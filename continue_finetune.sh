#!/bin/bash

# Initial settings
INITIAL_CKPT=  # Leave empty to start from scratch, or set to an initial checkpoint (e.g., 'exp/.../ckpt_epochX.pth')
INITIAL_EPOCHS=0  # Set to the epoch of the initial checkpoint (e.g., 150 if resuming from ckpt_epoch150.pth), 0 if starting fresh
EPOCHS_PER_PARTITION=150  # Number of epochs to train on each partition
BASE_EXP_DIR="exp/deepmar_resnet50/peta"

# Loop over partitions 0 to 4
for partition in {0..4}
do
    echo "Finetuning on partition $partition..."

    # Calculate total epochs for this partition
    TOTAL_EPOCHS=$((INITIAL_EPOCHS + (partition + 1) * EPOCHS_PER_PARTITION))

    # Determine checkpoint file
    if [ $partition -eq 0 ] && [ -z "$INITIAL_CKPT" ]; then
        # Start from scratch for partition 0 if no initial checkpoint
        CKPT_FILE=""
        RESUME="False"
    else
        # Use previous partition's checkpoint or initial checkpoint for partition 0
        if [ $partition -eq 0 ]; then
            CKPT_FILE="$INITIAL_CKPT"
        else
            PREV_PARTITION=$((partition - 1))
            CKPT_FILE="$BASE_EXP_DIR/partition$PREV_PARTITION/run1/model/ckpt_epoch$((INITIAL_EPOCHS + partition * EPOCHS_PER_PARTITION)).pth"
        fi
        RESUME="True"
    fi

    # Run training
    python script/experiment/train_deepmar_resnet50.py \
        --sys_device_ids="(0,)" \
        --dataset=peta \
        --partition_idx=$partition \
        --split=trainval \
        --test_split=test \
        --batch_size=32 \
        --resize="(224,224)" \
        --exp_subpath=deepmar_resnet50 \
        --new_params_lr=0.001 \
        --finetuned_params_lr=0.001 \
        --staircase_decay_at_epochs="(50,100)" \
        --total_epochs=$TOTAL_EPOCHS \
        --epochs_per_val=2 \
        --epochs_per_save=50 \
        --drop_pool5=True \
        --drop_pool5_rate=0.5 \
        --run=1 \
        --resume=$RESUME \
        --ckpt_file="$CKPT_FILE" \
        --test_only=False \
        --model_weight_file=

    # Check if training failed
    if [ $? -ne 0 ]; then
        echo "Training failed on partition $partition. Exiting."
        exit 1
    fi
done

echo "Finetuning completed across all 5 partitions."