from os import path
import yaml
import tempfile

from clearml import Task,Dataset,OutputModel,backend_api


task = Task.current_task()
params = task.get_parameters()
model_variant = params["Args/model_variant"]
tain_ds_id = params["Args/ds_id"]
epochs = int(params["Args/epochs"])


train_ds = Dataset.get(
        dataset_id=tain_ds_id
)
train_ds_dir = train_ds.get_local_copy()

data_config = {
    'path': train_ds_dir,
    'train': 'train',
    'val': 'validation',
    'test': 'test',
    'names': {0: "alpaca"}
}

# 保存字典为临时 YAML 文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
    yaml.dump(data_config, temp_file)
    temp_yaml_path = temp_file.name

args = dict(data=temp_yaml_path, epochs=epochs)

from ultralytics import YOLO
model = YOLO(f"{model_variant}.yaml")  # build a new model from scratch

results = model.train(**args)
print("Training completed results:", results)

# # 提取权重文件路径
# weights_dir = results.save_dir / 'weights'
# best_weights = weights_dir / 'best.pt'
# last_weights = weights_dir / 'last.pt'

local_dir = "runs/detect/train/weights/"
model_file_name = "best"

# upload_storage_uri

upload_uri = backend_api.Session.get_files_server_host();
print(f"PyTorch model uploaded to {upload_uri}")

# uploaded_model_uri = task.update_output_model(model_path=f"{local_dir}{model_file_name}.pt", model_name=f"{task.name}-pt")

# # 上传训练完成的 PyTorch 模型
# output_model = OutputModel(task=task,name=f"{task.name}-pt", comment="PyTorch", framework="PyTorch")
# uploaded_model_uri = output_model.update_weights(weights_filename=f'{model_file_name}.pt')
# print(f"PyTorch model uploaded to {uploaded_model_uri}")
# output_model.publish()  # 可选：将 PyTorch 模型发布



# 转换模型为 ONNX 格式
model.export(format="onnx", dynamic=True, opset=16)  # 导出 ONNX 模型
onnx_name = f"{model_file_name}.onnx"
# script_dir = path.dirname(path.abspath(__file__))
onnx_path = path.join(local_dir, onnx_name)
# remote_dir_path = uploaded_model_uri.rsplit('/', 1)[0]  # 移除最后的文件部分
# upload_uri = f"{remote_dir_path}/{onnx_name}"

# print(f"ONNX model exported to {onnx_path}")

# 上传 ONNX 模型到 ClearML
# uploaded_onnx_model_uri = task.update_output_model(model_path=onnx_path, model_name=f"{task.name}-onnx")
output_model_onnx = OutputModel(task=task,name=f"{task.name}-onnx",comment="ONNX", framework="ONNX")
uploaded_onnx_model_uri = output_model_onnx.update_weights(
    weights_filename=onnx_name,
    upload_uri=upload_uri,
    target_filename=onnx_name
    )
print(f"ONNX model uploaded to {uploaded_onnx_model_uri}")
# output_model_onnx.publish()  # 可选：将 ONNX 模型发布

# 上传 ONNX 模型作为任务的附加 Artifact
task.upload_artifact(name="onnx_model", artifact_object=onnx_path)

print("Training and model export complete. Models uploaded to ClearML.")


