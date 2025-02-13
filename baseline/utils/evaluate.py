import os
import torch
import numpy as np
import time

def extract_feat(feat_func, dataset, **kwargs):
    """
    Extract features for images.
    """
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True
    )

    start_time = time.time()
    total_eps = len(test_loader)
    N = len(dataset.image)
    start = 0

    # We'll store the final feature array in `feat`.
    # We need to know the shape from the first batch in order to initialize it.
    feat = None

    # Disable gradient tracking during feature extraction
    with torch.no_grad():
        for ep, (imgs, labels) in enumerate(test_loader):
            # Move images to GPU (if your model expects GPU inputs)
            imgs_var = imgs.cuda()

            # Pass through the feature function
            feat_tmp = feat_func(imgs_var)

            # If feat_tmp is a PyTorch tensor, convert to NumPy
            # If feat_tmp is already a NumPy array, do nothing
            if isinstance(feat_tmp, torch.Tensor):
                feat_tmp_np = feat_tmp.cpu().numpy()
            elif isinstance(feat_tmp, np.ndarray):
                feat_tmp_np = feat_tmp
            else:
                raise TypeError(
                    "feat_func returned an unsupported type: {}"
                    .format(type(feat_tmp))
                )

            batch_size = feat_tmp_np.shape[0]

            # If this is the first batch, initialize the 'feat' array
            if ep == 0:
                # Determine the flattened feature dimension after the first dimension (the batch).
                # E.g. if feat_tmp_np is (batch_size, C), dimension = C
                # If feat_tmp_np is (batch_size, C, H, W), dimension = C*H*W, etc.
                flat_dim = int(np.prod(feat_tmp_np.shape[1:]))
                feat = np.zeros((N, flat_dim), dtype=np.float32)

            # Reshape the feature array for this batch to 2D
            feat_2d = feat_tmp_np.reshape(batch_size, -1)

            # Place it in the correct spot of our preallocated `feat`
            feat[start:start+batch_size, :] = feat_2d
            start += batch_size

    end_time = time.time()
    print('{} batches done, total {:.2f}s'.format(total_eps, end_time - start_time))
    return feat


def attribute_evaluate(feat_func, dataset, **kwargs):
    """
    Perform attribute recognition evaluation.
    """
    print("Extracting features for attribute recognition")
    pt_result = extract_feat(feat_func, dataset)

    print("Computing attribute recognition results")
    N = pt_result.shape[0]
    L = pt_result.shape[1]

    # Prepare ground truth array
    gt_result = np.zeros(pt_result.shape, dtype=np.float32)
    for idx, label in enumerate(dataset.label):
        gt_result[idx, :] = label

    # Binarize predictions (>= 0 => 1, < 0 => 0)
    pt_result[pt_result >= 0] = 1
    pt_result[pt_result < 0] = 0

    return attribute_evaluate_lidw(gt_result, pt_result)


def attribute_evaluate_lidw(gt_result, pt_result):
    """
    Evaluate attribute predictions in both label-based and instance-based manners.
    """
    if gt_result.shape != pt_result.shape:
        print('Shape between groundtruth and predicted results are different.')
        raise ValueError

    result = {}

    # -- Label-based accuracy --
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)

    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)

    label_pos_acc = 1.0 * pt_pos / gt_pos
    label_neg_acc = 1.0 * pt_neg / gt_neg
    label_acc = (label_pos_acc + label_neg_acc) / 2.0

    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc

    # -- Instance-based metrics --
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1) + (pt_result == 1)).astype(float), axis=1)

    cnt_eff = float(gt_result.shape[0])
    for i, key in enumerate(gt_pos):
        # If ground-truth has zero positive labels,
        # avoid division by zero by artificially setting denominators to 1
        if key == 0:
            union_pos[i] = 1
            pt_pos[i] = 1
            gt_pos[i] = 1
            cnt_eff -= 1
            continue
        # Avoid dividing by zero if pt_pos[i] == 0
        if pt_pos[i] == 0:
            pt_pos[i] = 1

    instance_acc = np.sum(floatersect_pos / union_pos) / cnt_eff
    instance_precision = np.sum(floatersect_pos / pt_pos) / cnt_eff
    instance_recall = np.sum(floatersect_pos / gt_pos) / cnt_eff
    floatance_F1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)

    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1

    return result
