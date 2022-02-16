python object_detection/yolov5/train.py \
    --img 1280 --batch 8 \
    --epochs 500 --patience 100 \
    --label-smoothing 0.0 \
    --data data/AFO/dataset.yaml \
    --hyp object_detection/yolov5/data/hyps/hyp.scratch.yaml \
    --workers 16 \
    --project object_detection_ --name 027_yolov5_from_scratch
    # --weights yolov5s.pt \
