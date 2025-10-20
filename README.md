`python gradient_boosting_segmentation.py --csv BUSI_PROCESSED/dataset_manifest.csv \` <br>
`                                          --pretrained-checkpoint checkpoints/best.pth \ `<br>
`                                          --use-pretrained-as-first-booster \` <br>
`                                          --pretrained-booster-mode direct \` <br>
`                                          --weak-learner shallow-unet \` <br>
`                                          --num-boosters 6 \` <br>
`                                          --boosting-epochs-per-stage 20 \` <br>
`                                          --batch-size 2 \` <br>
`                                          --img-size 512 \` <br>
`                                          --base-channels 16 \` <br>
`                                          --num-workers 1 \` <br>
`                                          --gradient-accumulation-steps 4 \` <br>
`                                          --mixed-precision --validation-frequency 2 \` <br>
`                                          --disable-augmentation --pin-memory --lr 5e-4 \` <br>
`                                          --add-booster-threshold 0.03` <br>
