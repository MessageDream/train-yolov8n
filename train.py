from ultralytics import YOLO
from clearml import Task,Dataset,OutputModel

task = Task.get_task(project_name='DevOps', task_name='Yolo8n Remote Training')
output_model = OutputModel(task=task, framework="PyTorch")
params = task.get_parameters()
model_variant = params["model_variant"]
tain_ds_id = params["tain_ds_id"]
val_ds_id = params["val_ds_id"]
epochs = params["epochs"]

model = YOLO(f"{model_variant}.yaml")  # build a new model from scratch

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

args = dict(data=data_config, epochs=epochs)

results = model.train(**args)


