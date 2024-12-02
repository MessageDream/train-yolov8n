import yaml
import tempfile

from clearml import Task,Dataset,OutputModel


task = Task.get_task(project_name='DevOps', task_name='Yolo8n Remote Training')
output_model = OutputModel(task=task, framework="PyTorch")
params = task.get_parameters()
model_variant = params["Args/model_variant"]
tain_ds_id = params["Args/tain_ds_id"]
val_ds_id = params["Args/val_ds_id"]
epochs = int(params["Args/epochs"]) if params["Args/epochs"] else params["Args/epochs"]


train_ds = Dataset.get(
        dataset_id=tain_ds_id
)
train_ds_dir = train_ds.get_local_copy()

val_ds = Dataset.get(
        dataset_id=val_ds_id
)
val_ds_dir = val_ds.get_local_copy()

data_config = {
    'train': train_ds_dir,   # 训练图片的路径
    'val': val_ds_dir,       # 验证图片的路径
    'nc': 1,                 # 类别数量
    'names': ['alpaca']      # 类别名称
}

# 保存字典为临时 YAML 文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
    yaml.dump(data_config, temp_file)
    temp_yaml_path = temp_file.name

args = dict(data=temp_yaml_path, epochs=epochs)

from ultralytics import YOLO
model = YOLO(f"{model_variant}.yaml")  # build a new model from scratch

results = model.train(**args)


