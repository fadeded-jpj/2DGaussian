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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means2D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rots,
	const torch::Tensor& negative,
    const int image_height,
    const int image_width,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{

  const int P = means2D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means2D.options().dtype(torch::kInt32);
  auto float_opts = means2D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);

  torch::Tensor radii = torch::full({P}, 0, means2D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P,
		background.contiguous().data<float>(),
		W, H,
		means2D.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(),
		opacity.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		rots.contiguous().data_ptr<float>(),
		negative.contiguous().data_ptr<float>(),
		prefiltered,
		out_color.contiguous().data<float>(),
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rots,
	const torch::Tensor& negative,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means2D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  torch::Tensor dL_dconic = torch::zeros({P, 3}, means2D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 2}, means2D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means2D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means2D.options());
  torch::Tensor dL_drots = torch::zeros({P, 1}, means2D.options());
  torch::Tensor dL_dnega = torch::zeros({P, 1}, means2D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means2D.options());




  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::backward(P, R,
	  background.contiguous().data<float>(),
	  W, H,
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.contiguous().data_ptr<float>(),
	  rots.contiguous().data_ptr<float>(),
	  negative.contiguous().data_ptr<float>(),
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drots.contiguous().data_ptr<float>(),
	  dL_dnega.contiguous().data_ptr<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dscales, dL_drots, dL_dnega);
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({2 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}