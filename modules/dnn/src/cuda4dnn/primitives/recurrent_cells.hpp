// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CELLS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CELLS_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/tensor_ops.hpp"
#include "../csl/cudnn/recurrent.hpp"

#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn {

struct RNNConfiguration
{
    int seqLength;
    int numLayers;
    int hiddenSize;
    int inputSize;
    int miniBatch;
    bool bidirectional;
};

template <class T, class Enable = void>
struct LSTMFilterPacker
{
    static Mat pack(const csl::cudnn::Handle&, const Mat& filters, const RNNConfiguration&)
    {
        return filters;
    }
};

inline Mat packLSTMFiltersFP32(const csl::cudnn::Handle& handle, const Mat& filters, const RNNConfiguration& config)
{
    CV_Assert(filters.type() == CV_32F);
    Mat contiguous = filters.isContinuous() ? filters : filters.clone();

    csl::cudnn::DropoutDescriptor dropoutDesc(handle, 0.0f);
    csl::cudnn::RNNDescriptor<float> rnnDesc(handle,
        csl::cudnn::RNNDescriptor<float>::RNNMode::LSTM,
        config.hiddenSize, config.numLayers, config.bidirectional, dropoutDesc);
    csl::cudnn::TensorDescriptor<float> xDesc({config.miniBatch, config.inputSize, 1});

    size_t sizeInBytes = 0;
    CUDA4DNN_CHECK_CUDNN(cudnnGetRNNParamsSize(handle.get(), rnnDesc.get(), xDesc.get(),
                                               &sizeInBytes, CUDNN_DATA_FLOAT));
    CV_Assert(sizeInBytes / sizeof(float) == contiguous.total());

    Mat packed = Mat::zeros(contiguous.size(), contiguous.type());
    const float* src = contiguous.ptr<float>();
    float* dst = packed.ptr<float>();
    csl::cudnn::FilterDescriptor<float> weightsDesc({static_cast<int>(contiguous.total()), 1, 1});

    const int numDirections = config.bidirectional ? 2 : 1;
    const size_t wxStride = static_cast<size_t>(4) * config.hiddenSize * config.inputSize;
    const size_t whStride = static_cast<size_t>(4) * config.hiddenSize * config.hiddenSize;
    const size_t bStride = static_cast<size_t>(8) * config.hiddenSize;
    const size_t dirStride = wxStride + whStride + bStride;

    for (int pseudoLayer = 0; pseudoLayer < config.numLayers * numDirections; ++pseudoLayer)
    {
        const float* dirBase = src + pseudoLayer * dirStride;
        const float* wxBase = dirBase;
        const float* whBase = wxBase + wxStride;
        const float* wbBase = whBase + whStride;
        const float* rbBase = wbBase + static_cast<size_t>(4) * config.hiddenSize;

        for (int linLayerID = 0; linLayerID < 8; ++linLayerID)
        {
            cudnnFilterDescriptor_t matDesc = nullptr;
            void* matPtr = nullptr;
            CUDA4DNN_CHECK_CUDNN(cudnnCreateFilterDescriptor(&matDesc));
            CUDA4DNN_CHECK_CUDNN(cudnnGetRNNLinLayerMatrixParams(handle.get(), rnnDesc.get(),
                pseudoLayer, xDesc.get(), weightsDesc.get(), dst,
                linLayerID, matDesc, &matPtr));
            const size_t matOffset = static_cast<float*>(matPtr) - dst;
            const size_t matSize = (linLayerID < 4)
                ? static_cast<size_t>(config.hiddenSize) * config.inputSize
                : static_cast<size_t>(config.hiddenSize) * config.hiddenSize;
            const float* matSrc = (linLayerID < 4)
                ? wxBase + static_cast<size_t>(linLayerID) * config.hiddenSize * config.inputSize
                : whBase + static_cast<size_t>(linLayerID - 4) * config.hiddenSize * config.hiddenSize;
            std::copy(matSrc, matSrc + matSize, dst + matOffset);
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyFilterDescriptor(matDesc));

            cudnnFilterDescriptor_t biasDesc = nullptr;
            void* biasPtr = nullptr;
            CUDA4DNN_CHECK_CUDNN(cudnnCreateFilterDescriptor(&biasDesc));
            CUDA4DNN_CHECK_CUDNN(cudnnGetRNNLinLayerBiasParams(handle.get(), rnnDesc.get(),
                pseudoLayer, xDesc.get(), weightsDesc.get(), dst,
                linLayerID, biasDesc, &biasPtr));
            const size_t biasOffset = static_cast<float*>(biasPtr) - dst;
            const float* biasSrc = (linLayerID < 4)
                ? wbBase + static_cast<size_t>(linLayerID) * config.hiddenSize
                : rbBase + static_cast<size_t>(linLayerID - 4) * config.hiddenSize;
            std::copy(biasSrc, biasSrc + config.hiddenSize, dst + biasOffset);
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyFilterDescriptor(biasDesc));
        }
    }

    return packed;
}

template <class T>
struct LSTMFilterPacker<T, typename std::enable_if<std::is_same<T, float>::value>::type>
{
    static Mat pack(const csl::cudnn::Handle& handle, const Mat& filters, const RNNConfiguration& config)
    {
        return packLSTMFiltersFP32(handle, filters, config);
    }
};

template <class T>
struct LSTMFilterPacker<T, typename std::enable_if<std::is_same<T, half>::value>::type>
{
    static Mat pack(const csl::cudnn::Handle& handle, const Mat& filters, const RNNConfiguration& config)
    {
        return packLSTMFiltersFP32(handle, filters, config);
    }
};

template<class T>
class LSTMOp final : public CUDABackendNode
{
public:
    using wrapper_type = GetCUDABackendWrapperType<T>;

    LSTMOp(csl::Stream stream_, csl::cudnn::Handle handle, const Mat& filters, const Mat& h0,
           const Mat& c0, const RNNConfiguration& config)
            : stream(std::move(stream_))
    {
        typename csl::LSTM<T>::params_type params{
                {filters.total(), 1, 1}, // reshape
                config.seqLength,
                config.numLayers,
                config.hiddenSize,
                config.inputSize,
                config.miniBatch,
                config.bidirectional,
                0.0, /* dropout */
                csl::cudnn::RNNDescriptor<T>::RNNMode::LSTM
        };

        lstm = csl::LSTM<T>(handle, params);
        Mat packed_filters = LSTMFilterPacker<T>::pack(handle, filters, config);
        auto correct_shape_filters = packed_filters.reshape(1, {static_cast<int>(packed_filters.total()), 1, 1});
        filtersTensor = csl::makeTensorHeader<T>(correct_shape_filters);
        csl::copyMatToTensor<T>(correct_shape_filters, filtersTensor, stream);

        h0Tensor = csl::makeTensorHeader<T>(h0);
        csl::copyMatToTensor<T>(h0, h0Tensor, stream);

        c0Tensor = csl::makeTensorHeader<T>(c0);
        csl::copyMatToTensor<T>(c0, c0Tensor, stream);

        csl::WorkspaceBuilder builder;
        builder.require<T>(lstm.get_workspace_memory_in_bytes());
    }

    void forward(const std::vector<cv::Ptr<BackendWrapper>>& inputs,
                 const std::vector<cv::Ptr<BackendWrapper>>& outputs,
                 csl::Workspace& workspace) override
    {
        CV_Assert(inputs.size() == 1 && !outputs.empty());

        auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
        auto input = input_wrapper->getView();

        auto y_output_wrapper = outputs[0].dynamicCast<wrapper_type>();
        auto y_output = y_output_wrapper->getSpan();

        Ptr<wrapper_type> yc_output_wrapper = outputs.size() == 2 ? outputs[1].dynamicCast<wrapper_type>() : Ptr<wrapper_type>();
        csl::TensorSpan<T> yc_output = yc_output_wrapper.empty() ? csl::TensorSpan<T>() : yc_output_wrapper->getSpan();

        csl::WorkspaceAllocator allocator(workspace);
        lstm.inference(input, y_output, yc_output, filtersTensor, h0Tensor, c0Tensor, allocator.get_instance());
    }

    std::size_t get_workspace_memory_in_bytes() const noexcept override
    {
        return lstm.get_workspace_memory_in_bytes();
    }

private:
    csl::LSTM<T> lstm;
    csl::Stream stream;
    csl::Tensor<T> filtersTensor;
    csl::Tensor<T> h0Tensor;
    csl::Tensor<T> c0Tensor;
};

}}} /* namespace cv::dnn::cuda4dnn */

#endif //OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RECURRENT_CELLS_HPP