from os import path
import yaml
import tempfile

from clearml import Task,Dataset,OutputModel


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
    'names': ['alpaca']
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

# 上传训练完成的 PyTorch 模型
output_model = OutputModel(task=task,name=f"{task.name}-pt", comment="PyTorch", framework="PyTorch")
output_model.update_weights(weights_filename='best.pt')
# output_model.publish()  # 可选：将 PyTorch 模型发布

# 转换模型为 ONNX 格式
onnx_path = model.export(format="onnx", dynamic=True, opset=16)  # 导出 ONNX 模型
print(f"ONNX model exported to {onnx_path}")
onnx_name = path.basename(onnx_path)

# 上传 ONNX 模型到 ClearML
output_model_onnx = OutputModel(task=task,name=f"{task.name}_onnx",comment="ONNX", framework="ONNX")
output_model_onnx.update_weights(weights_filename=onnx_name)
# output_model_onnx.publish()  # 可选：将 ONNX 模型发布

# 上传 ONNX 模型作为任务的附加 Artifact
task.upload_artifact(name="onnx_model", artifact_object=onnx_path)

print("Training and model export complete. Models uploaded to ClearML.")


