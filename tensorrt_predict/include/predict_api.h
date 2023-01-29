#ifndef PREDICT_API_H_
#define PREDICT_API_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef _WIN32
#ifdef TRT_EXPORTS
#define TRT_DLL __declspec(dllexport)
#else
#define TRT_DLL __declspec(dllimport)
#endif
#else
#define TRT_DLL
#endif
#include <cuda_runtime_api.h>

/*! \brief manually define unsigned int */
typedef unsigned int trt_uint;
/*! \brief manually define float */
typedef float trt_float;
/*! \brief handle to Predictor */
typedef void *PredictorHandle;


/*!
 * \brief create a predictor
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_id The device id of the predictor.
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
TRT_DLL int InferenceCreate(const void *param_bytes,
                 int param_size,
                 int dev_id,
                 PredictorHandle *out);

/*!
 * \brief Get the shape of output node.
 *  The returned shape_data and shape_ndim is only valid before next call to MXPred function.
 * \param handle The handle of the predictor.
 * \param index The index of output node, set to 0 if there is only one output.
 * \param shape_data Used to hold pointer to the shape data
 * \param shape_ndim Used to hold shape dimension.
 * \return 0 when success, -1 when failure.
 */
TRT_DLL int PredGetOutputShape(PredictorHandle handle,
                         trt_uint index,
                         trt_uint **shape_data,
                         trt_uint *shape_ndim);

/*!
 * \brief Set the input data of predictor.
 * \param handle The predictor handle.
 * \param key The name of input node to set.
 *     For feedforward net, this is "data".
 * \param data The pointer to the data to be set, with the shape specified in MXPredCreate.
 * \param size The size of data array, used for safety check.
 * \return 0 when success, -1 when failure.
 */
TRT_DLL int PredSetInput(PredictorHandle handle,
                   const char *key,
                   const trt_float *data,
                   trt_uint size);


TRT_DLL int PredSetInputAuto(PredictorHandle handle);

/*!
 * \brief Run a forward pass to get the output.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
TRT_DLL int PredForward(PredictorHandle handle);

TRT_DLL int PredForwardAsync(PredictorHandle handle, cudaStream_t stream);
/*!
 * \brief Get the output value of prediction.
 * \param handle The handle of the predictor.
 * \param index The index of output node, set to 0 if there is only one output.
 * \param data User allocated data to hold the output.
 * \param size The size of data array, used for safe checking.
 * \return 0 when success, -1 when failure.
 */
TRT_DLL int PredGetOutput(PredictorHandle handle,
                              trt_uint index,
                              trt_float* data,
                              trt_uint size);

/*!
 * \brief Free a predictor handle.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
TRT_DLL int PredFree(PredictorHandle handle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // PREDICT_API_H_
