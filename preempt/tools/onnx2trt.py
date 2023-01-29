import os
import argparse


models = {
    # "fasterrn"
    "pointpillar": {
        "onnx": "pointpillar.onnx",
        "shapes": [
            "voxels:1x32x4,num_points:1,coors:1x4",
            "voxels:1000x32x4,num_points:1000,coors:1000x4",
            "voxels:10000x32x4,num_points:10000,coors:10000x4",
        ]
    },
    "deepsort": {
        "onnx": "deepsort.onnx",
        "shapes": [
            "input:1x3x128x64,output:1x512",
            "input:8x3x128x64,output:8x512",
            "input:64x3x128x64,output:64x512"
        ]
    },
    "yolov5l": {
        "onnx": "yolov5l.onnx",
        "shapes": []
    },
    "fasterrcnn-new": {
        "onnx": "fasterrcnn-new.onnx",
        "shapes": [
            "input:1x3x320x320",
            "input:1x3x720x1280",
            "input:1x3x1344x1344",
        ]
    },
    "resa_detection": {
        "onnx": "resa_detection.onnx",
        "shapes": []
    },
}

def onnx2trt(name, model, onnx_dir, plugin_path, trtexec_path):
    onnx_path = os.path.join(onnx_dir, model["onnx"])
    engine_path = os.path.join(onnx_dir, name+".engine")
    shape_str = ""
    if len(model["shapes"]) != 0:
        shape_str = f"--explicitBatch --minShapes={model['shapes'][0]} --optShapes={model['shapes'][1]} --maxShapes={model['shapes'][2]} --shapes={model['shapes'][1]}"
    assert os.system(f"{trtexec_path} --onnx={onnx_path} --saveEngine={engine_path} --plugins={plugin_path} --workspace=1024 {shape_str}") == 0
    

if __name__ == '__main__':
    """Transform DNN models from ONNX to TensorRT engine.
    """
    parser = argparse.ArgumentParser(description="Transform DNN models from ONNX to TensorRT engine.")
    parser.add_argument("model_dir", help="directory of the ONNX files")
    parser.add_argument("plugin_path", help="path of mmdepoly plugins")
    parser.add_argument("trtexec_path", help="path of trtexec tool")
    parser.add_argument("model_name", help="name of the model")

    args = parser.parse_args()

    if args.model_name == "all":
        for name in models:
            onnx2trt(name, models[name], args.model_dir, args.plugin_path, args.trtexec_path)
    else:
        onnx2trt(args.model_name, models[args.model_name], args.model_dir, args.plugin_path, args.trtexec_path)


