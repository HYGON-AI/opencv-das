/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../../precomp.hpp"
#include <string>
#include <vector>
#include "../include/common.hpp"
#include "../include/ocl4dnn.hpp"
#include "opencl_kernels_dnn.hpp"

#include <limits.h>
#include <stdint.h>

typedef uint16_t src_t;
typedef uint16_t src_rep_t;
#define SRC_REP_C UINT16_C
static const int srcSigBits = 10;
#define src_rep_t_clz __builtin_clz

typedef float dst_t;
typedef uint32_t dst_rep_t;
#define DST_REP_C UINT32_C
static const int dstSigBits = 23;

// End of specialization parameters.  Two helper routines for conversion to and
// from the representation of floating-point data as integer values follow.

static __inline src_rep_t srcToRep(src_t x) {
    const union { src_t f; src_rep_t i; } rep = {.f = x};
    return rep.i;
}

static __inline dst_t dstFromRep(dst_rep_t x) {
    const union { dst_t f; dst_rep_t i; } rep = {.i = x};
    return rep.f;
}
// End helper routines.  Conversion implementation follows.

static __inline dst_t __extendXfYf2__(src_t a) {
    // Various constants whose values follow from the type parameters.
    // Any reasonable optimizer will fold and propagate all of these.
    const int srcBits = sizeof(src_t)*CHAR_BIT;
    const int srcExpBits = srcBits - srcSigBits - 1;
    const int srcInfExp = (1 << srcExpBits) - 1;
    const int srcExpBias = srcInfExp >> 1;

    const src_rep_t srcMinNormal = SRC_REP_C(1) << srcSigBits;
    const src_rep_t srcInfinity = (src_rep_t)srcInfExp << srcSigBits;
    const src_rep_t srcSignMask = SRC_REP_C(1) << (srcSigBits + srcExpBits);
    const src_rep_t srcAbsMask = srcSignMask - 1;
    const src_rep_t srcQNaN = SRC_REP_C(1) << (srcSigBits - 1);
    const src_rep_t srcNaNCode = srcQNaN - 1;

    const int dstBits = sizeof(dst_t)*CHAR_BIT;
    const int dstExpBits = dstBits - dstSigBits - 1;
    const int dstInfExp = (1 << dstExpBits) - 1;
    const int dstExpBias = dstInfExp >> 1;

    const dst_rep_t dstMinNormal = DST_REP_C(1) << dstSigBits;

    // Break a into a sign and representation of the absolute value
    const src_rep_t aRep = srcToRep(a);
    const src_rep_t aAbs = aRep & srcAbsMask;
    const src_rep_t sign = aRep & srcSignMask;
    dst_rep_t absResult;

    // If sizeof(src_rep_t) < sizeof(int), the subtraction result is promoted
    // to (signed) int.  To avoid that, explicitly cast to src_rep_t.
    if ((src_rep_t)(aAbs - srcMinNormal) < srcInfinity - srcMinNormal) {
        // a is a normal number.
        // Extend to the destination type by shifting the significand and
        // exponent into the proper position and rebiasing the exponent.
        absResult = (dst_rep_t)aAbs << (dstSigBits - srcSigBits);
        absResult += (dst_rep_t)(dstExpBias - srcExpBias) << dstSigBits;
    }

    else if (aAbs >= srcInfinity) {
        // a is NaN or infinity.
        // Conjure the result by beginning with infinity, then setting the qNaN
        // bit (if needed) and right-aligning the rest of the trailing NaN
        // payload field.
        absResult = (dst_rep_t)dstInfExp << dstSigBits;
        absResult |= (dst_rep_t)(aAbs & srcQNaN) << (dstSigBits - srcSigBits);
        absResult |= (dst_rep_t)(aAbs & srcNaNCode) << (dstSigBits - srcSigBits);
    }

    else if (aAbs) {
        // a is denormal.
        // renormalize the significand and clear the leading bit, then insert
        // the correct adjusted exponent in the destination type.
        const int scale = src_rep_t_clz(aAbs) - src_rep_t_clz(srcMinNormal);
        absResult = (dst_rep_t)aAbs << (dstSigBits - srcSigBits + scale);
        absResult ^= dstMinNormal;
        const int resultExponent = dstExpBias - srcExpBias - scale + 1;
        absResult |= (dst_rep_t)resultExponent << dstSigBits;
    }

    else {
        // a is zero.
        absResult = 0;
    }

    // Apply the signbit to (dst_t)abs(a).
    const dst_rep_t result = absResult | (dst_rep_t)sign << (dstBits - srcBits);
    return dstFromRep(result);
}
// Use a forwarding definition and noinline to implement a poor man's alias,
// as there isn't a good cross-platform way of defining one.
__attribute__((noinline)) float __extendhfsf2(uint16_t a) {
    return __extendXfYf2__(a);
}

extern "C" float __gnu_h2f_ieee(uint16_t a) {
    return __extendhfsf2(a);
}

namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNPool<Dtype>::OCL4DNNPool(OCL4DNNPoolConfig config)
{
    int dims = config.in_shape.size();
    int spatial_dims = config.in_shape.size()-2;

    channels_ = config.channels;
    pool_method_ = config.pool_method;
    avePoolPaddedArea = config.avePoolPaddedArea;
    computeMaxIdx = config.computeMaxIdx;
    use_half = config.use_half;
    kernel_shape_.push_back(config.kernel.height);
    kernel_shape_.push_back(config.kernel.width);
    stride_.push_back(config.stride.height);
    stride_.push_back(config.stride.width);

    for (int i = 0; i < spatial_dims; ++i)
    {
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    kernel_h_ = kernel_shape_[0];
    kernel_w_ = kernel_shape_[1];
    stride_h_ = stride_[0];
    stride_w_ = stride_[1];
    pad_t_ = config.pad_t;
    pad_l_ = config.pad_l;
    pad_r_ = config.pad_r;
    pad_b_ = config.pad_b;
    height_ = spatial_dims == 1? 1 : im_in_shape_[0];
    width_ = im_in_shape_.back();
    pooled_height_ = spatial_dims == 1? 1 : im_out_shape_[0];
    pooled_width_ = im_out_shape_.back();

    count_ = 1;
    for (int i = 0; i < config.out_shape.size(); ++i)
    {
        count_ *= config.out_shape[i];
    }
}

template<typename Dtype>
OCL4DNNPool<Dtype>::~OCL4DNNPool()
{
    // nothing
}

template<typename Dtype>
bool OCL4DNNPool<Dtype>::Forward(const UMat& bottom,
                                 UMat& top,
                                 UMat& top_mask)
{
    bool ret = true;
    size_t global[] = { (size_t)count_ };
    size_t local[] = { 128 };

    // support 2D case
    switch (pool_method_)
    {
    case LIBDNN_POOLING_METHOD_MAX:
        {
            String kname = computeMaxIdx ? "max_pool_forward_mask" : "max_pool_forward";
            kname += (use_half) ? "_half" : "_float";
            ocl::Kernel oclk_max_pool_forward(
                kname.c_str(),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format(" -D Dtype=%s -D KERNEL_MAX_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d"
                       " -D PAD_L=%d -D PAD_T=%d -D PAD_R=%d -D PAD_B=%d%s",
                       (use_half) ? "half" : "float",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_,
                       pad_l_, pad_t_, pad_r_, pad_b_,
                       computeMaxIdx ? " -D HAVE_MASK=1" : ""
                ));
            if (oclk_max_pool_forward.empty())
                return false;

            oclk_max_pool_forward.args(
                count_,
                ocl::KernelArg::PtrReadOnly(bottom),
                channels_,
                height_,
                width_,
                pooled_height_,
                pooled_width_,
                ocl::KernelArg::PtrWriteOnly(top)
            );
            if (computeMaxIdx)
                oclk_max_pool_forward.set(8, ocl::KernelArg::PtrWriteOnly(top_mask));  // TODO remove magic number. Extend cv::ocl::Kernel API

            ret = oclk_max_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_AVE:
        {
            CV_Assert(top_mask.empty());

            String kname = format("ave_pool_forward_%s", (use_half) ? "half" : "float");
            ocl::Kernel oclk_ave_pool_forward(
                kname.c_str(),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format(" -D Dtype=%s -D KERNEL_AVE_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d"
                       " -D PAD_L=%d -D PAD_T=%d -D PAD_R=%d -D PAD_B=%d%s",
                       (use_half) ? "half" : "float",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_,
                       pad_l_, pad_t_, pad_r_, pad_b_,
                       avePoolPaddedArea ? " -D AVE_POOL_PADDING_AREA" : ""
                ));

            if (oclk_ave_pool_forward.empty())
                return false;

            oclk_ave_pool_forward.args(
                count_,
                ocl::KernelArg::PtrReadOnly(bottom),
                channels_,
                height_,
                width_,
                pooled_height_,
                pooled_width_,
                ocl::KernelArg::PtrWriteOnly(top)
            );

            ret = oclk_ave_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_STO:
        {
            CV_Assert(top_mask.empty());

            String kname = format("sto_pool_forward_test_%s", (use_half) ? "half" : "float");
            ocl::Kernel oclk_sto_pool_forward(
                kname.c_str(),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format(" -D Dtype=%s -D KERNEL_STO_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d",
                       (use_half) ? "half" : "float",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_
                ));


            if (oclk_sto_pool_forward.empty())
                return false;

            oclk_sto_pool_forward.args(
                count_,
                ocl::KernelArg::PtrReadOnly(bottom),
                channels_,
                height_,
                width_,
                pooled_height_,
                pooled_width_,
                ocl::KernelArg::PtrWriteOnly(top)
            );

            ret = oclk_sto_pool_forward.run(1, global, local, false);
        }
        break;
    default:
        {
            ret = false;
            LOG(FATAL)<< "Unknown pooling method.";
        }
    }
    return ret;
}

template class OCL4DNNPool<float>;

}}} // namespace cv::dnn::ocl4dnn
