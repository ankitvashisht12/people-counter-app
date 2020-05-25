# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

While converting pre-trained model into IR, Model Optimizer searches for each layer of the input model in the list of known layers.The list of known layers is different for each of supported frameworks.

Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

When implementing a custom layer for the model, we need to add extensions to both the Model Optimizer and the Inference Engine.While loading IR in inference Engine, if we encounter unsupported layers then we should use cpu extension

> Note: The primary CPU extension file differs in naming between Linux and Mac. On Linux, the name is libcpu_extension_sse4.so, while on Mac it is libcpu_extension.dylib.

Some of the potential reasons for handling custom layers are...

* We need to handle custom layers otherwise model optimizer can not convert pre-trained model into IR format.


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

##### Accuracy Comparison
The difference between model accuracy pre- and post-conversion was...
Accuracy of the pre-conversion model and post-conversion both are approx. same and good.

##### Size Comparison
The size of the model pre- and post-conversion was...
Size of the pre-conversion model i.e. `fozen inference graph(.pb file) = 69.7 MB` and size of the pos-conversion model i.e. `xml+bin file = 67.5 MB`

##### Inference time Comparison
The inference time of the model pre- and post-conversion was...
Average Inference time of post conversion model is approx. 75 times more than pre conversion model.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1. People counter app can be used in Malls and grocery stores to do comparison analysis between Malls about their popularity. More people visits means more popularity.

2. It can be used to analyze the duration between two people thus can be deployed worldwide during COVID-19 pandemic.

3. It can be used to monitor people in restricted areas.

Each of these use cases would be useful because...
1. It gives popularity statistics.
2. Helps in enforcing social distancing in global pandemic like COVID-19.
3. Helps in preventing intrusions in restricted areas.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

1. Lighting : Lighting can effects the accuracy of model prediction as low lighting condition makes difficult to detect person in the frame.
2. Model accuracy : Model accuracy is most crucial factor as where higher model accuracy is favorable, lower model accuracy can have false results.
3. Focal length : Focal length doesn't have much effect and totally depends on use case or requirement of users.
4. Image Size : Image Size can have great effect as higher resolution image can potentially give correct result.

## Model Used 

* Download the model from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

## Command Used to Convert .pb file into IR 

I used below command to convert frozen inference graph (.pb) file into Intermediate Representation (xml + bin)

```py
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel
```