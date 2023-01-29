
#ifndef TENSORRT_PREDICT_PLUGIN_FACTORY_H
#define TENSORRT_PREDICT_PLUGIN_FACTORY_H
#include <NvInfer.h>
#include <NvUtils.h>
#include <NvInferPlugin.h>
#include <vector>

namespace trt_predict
{

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory {
 public:
  static const char *plugin_reshape_key_;

  static const char *plugin_rproi_key_;

  static const char *plugin_concat_key_;

  // deserialization plugin implementation
  nvinfer1::IPlugin *createPlugin(const char *layerName,
                                  const void *serialData,
                                  size_t serialLength) override;

  // plugin implementation
  bool isPlugin(const char *name);

  // the application has to destroy the plugin when it knows it's safe to do so
  void destroyPlugin();

 private:
  std::vector<nvinfer1::IPlugin *> plugin_pools_;

  std::vector<nvinfer1::plugin::INvPlugin *> nv_plugin_pools_;
};

} // end namespace trt_predict

#endif
