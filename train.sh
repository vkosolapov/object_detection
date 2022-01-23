python object_detection/yolov5/train.py \
    --img 1280 --batch 8 \
    --epochs 500 --patience 25 \
    --label-smoothing 0.01 \
    --data data/AFO/dataset.yaml \
    --weights yolov5s.pt \
    --hyp object_detection/yolov5/data/hyps/hyp.scratch.yaml \
    --workers 12 \
    --project object_detection_ --name 011_soft_loss_and_aug
