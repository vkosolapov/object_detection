python object_detection/yolov5/train.py \
    --img 1280 --batch 8 \
    --epochs 500 --patience 100 \
    --label-smoothing 0.0 \
    --optimizer Ranger \
    --data data/AFO/dataset.yaml \
    --weights yolov5s.pt \
    --hyp object_detection/yolov5/data/hyps/hyp.scratch.yaml \
    --workers 16 \
    --project object_detection_ --name 014_no_aug_loss_change
