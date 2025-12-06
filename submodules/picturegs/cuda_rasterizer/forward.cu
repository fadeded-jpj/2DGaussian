/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include<iostream>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__device__ void computeCov2D(const glm::vec2 scale, const float rot, float* cov2D)
{
	glm::mat2 S = glm::mat2(0.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;

	glm::mat2 R = glm::mat2(
		glm::cos(rot),  glm::sin(rot),
		-1.0f *glm::sin(rot), glm::cos(rot)
	);

	glm::mat2 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov2D[0] = Sigma[0][0];
	cov2D[1] = Sigma[0][1];
	cov2D[2] = Sigma[1][1];
}

template<int C>
__global__ void preprocessCUDA(int P,
	const float* D2D,
	const glm::vec2* scales,
	const float* rots,
	const float* negatives,
	const float* opacities,
	const float* colors,
	const int W, int H,
	int* radii,
	float* cov2Ds,
	float2* points_xy_image,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	computeCov2D(scales[idx], rots[idx], cov2Ds + idx * 3);
	float3 cov = { cov2Ds[3*idx+0], cov2Ds[3*idx+1],cov2Ds[3*idx+2]};
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	const float det = det_cov;

	if (det_cov == 0.0f)
	return;
	float det_inv = 1.f / det;

	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };







	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = {D2D[2 * idx], D2D[2 * idx+1] };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.

	// Store some useful helper data for the next steps.
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];
    // if (idx < 10) {  // 打印前10个线程
    //     printf("[Thread %d] opacity = %f\n", idx, opacity);
    // }
	conic_opacity[idx].w=opacity;
	conic_opacity[idx].x=conic.x;
	conic_opacity[idx].y=conic.y;
	conic_opacity[idx].z=conic.z;

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	//     if (idx < 5) {
    //     printf("GPU [idx=%d]: scale=(%.6f,%.6f), rot=%.6f\n", 
    //            idx, scales[idx].x, scales[idx].y, rots[idx]);
    //     printf("GPU [idx=%d]: cov2D=(%.6f,%.6f,%.6f)\n", 
    //            idx, cov2Ds[0], cov2Ds[1], cov2Ds[2]);
    //     printf("GPU [idx=%d]: det=%.6f, det_inv=%.6f\n", 
    //            idx, det, det_inv);
    //     printf("GPU [idx=%d]: conic=(%.6f,%.6f,%.6f), opacity=%.6f\n", 
    //            idx, conic.x, conic.y, conic.z, conic_opacity[idx].w);
    // }
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(const float* negative,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power))*negative[collected_id[j]];
			if (alpha < 1.0f / 255.0f)
				continue;
			// float test_T = T - alpha;

			// if (test_T < 0.0001f)
			// {
			// 	done = true;
			// 	continue;
			// }
			// T = test_T;

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha;




			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			if(j== min(BLOCK_SIZE, toDo)){
				done=true;

			}
			
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch];

	}
}

void FORWARD::render(const float* negative,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (negative,
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, const float* D2D,
	const glm::vec2* scales,
	const float* rots,
	const float* negatives,
	const float* opacities,
	const float* colors,
	const int W, int H,
	int* radii,
	float* cov2Ds,
	float2* means2D,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,D2D,
		scales,
		rots,
		negatives,
		opacities,
		colors,
		W, H,
		radii,
		cov2Ds,
		means2D,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
