`python gradient_boosting_segmentation.py --csv BUSI_PROCESSED/dataset_manifest.csv \` <br>
`                                          --pretrained-checkpoint checkpoints/best.pth \ `
`                                          --use-pretrained-as-first-booster \`
`                                          --pretrained-booster-mode direct \`
`                                          --weak-learner shallow-unet \`
`                                          --num-boosters 6 \`
`                                          --boosting-epochs-per-stage 20 \`
`                                          --batch-size 2 \`
`                                          --img-size 512 \`
`                                          --base-channels 16 \`
`                                          --num-workers 1 \`
`                                          --gradient-accumulation-steps 4 \`
`                                          --mixed-precision --validation-frequency 2 \`
`                                          --disable-augmentation --pin-memory --lr 5e-4 \`
`                                          --add-booster-threshold 0.03`
