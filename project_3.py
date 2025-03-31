import imageio as iio
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import Axes3D

vol_d = iio.volread('/kaggle/input/cardiac-ct-dicom-dataset/Diastolic1', 'DICOM')
start_slice_d = 85
end_slice_d = 140

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(vol_d[start_slice_d], cmap='gray')
axes[0].set_title(f"Diastolic Start Slice ({start_slice_d})")
axes[0].axis('off')

axes[1].imshow(vol_d[end_slice_d], cmap='gray')
axes[1].set_title(f"Diastolic End Slice ({end_slice_d})")
axes[1].axis('off')

plt.show()

volume_d = np.zeros(end_slice_d - start_slice_d + 1)

d0_d, d1_d, d2_d = vol_d.meta['sampling']
dvoxel_d = d0_d * d1_d * d2_d  

import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def segment_left_ventricle(image, phase='diastole', slice_num=None, 
                          lower_thresh=250, upper_thresh=400, filter_sigma=1):
    """
    Enhanced cardiac LV segmentation with phase-first parameter selection
    and improved slice-specific handling
    """
    if phase == 'diastole':
        base_marker_size = 12
        base_lower_thresh = lower_thresh
        base_upper_thresh = upper_thresh
        
        y_offset = 0
    else:  
        base_marker_size = 10  
        base_lower_thresh = lower_thresh - 10  
        base_upper_thresh = upper_thresh + 20  
        y_offset = -8  
    
    if slice_num is not None:
        if slice_num <= 95:  
            if phase == 'diastole':
                local_lower = base_lower_thresh
                local_upper = base_upper_thresh
                marker_size = base_marker_size
                lv_y, lv_x = 220 + y_offset, 290
                la_y, la_x = 260 + y_offset, 240
            else: 
                local_lower = base_lower_thresh - 40  
                local_upper = base_upper_thresh + 40  
                marker_size = base_marker_size + 6  
                lv_y, lv_x = 210 + y_offset, 280  
                la_y, la_x = 250 + y_offset, 235
                
        elif slice_num < 100:  
            if phase == 'diastole':
                local_lower = base_lower_thresh
                local_upper = base_upper_thresh
                marker_size = base_marker_size
                lv_y, lv_x = 220 + y_offset, 290
                la_y, la_x = 260 + y_offset, 240
            else: 
                local_lower = base_lower_thresh - 35
                local_upper = base_upper_thresh + 35
                marker_size = base_marker_size + 5
                lv_y, lv_x = 212 + y_offset, 282
                la_y, la_x = 252 + y_offset, 238
                
        elif slice_num < 105:  
            if phase == 'diastole':
                local_lower = base_lower_thresh
                local_upper = base_upper_thresh
                marker_size = base_marker_size
                lv_y, lv_x = 225 + y_offset, 300
                la_y, la_x = 260 + y_offset, 250
            else:  
                local_lower = base_lower_thresh - 25
                local_upper = base_upper_thresh + 30
                marker_size = base_marker_size + 3
                lv_y, lv_x = 215 + y_offset, 290
                la_y, la_x = 250 + y_offset, 242
                
        elif slice_num < 110:  
            if phase == 'diastole':
                local_lower = base_lower_thresh
                local_upper = base_upper_thresh
                marker_size = base_marker_size
                lv_y, lv_x = 225 + y_offset, 300
                la_y, la_x = 260 + y_offset, 250
            else: 
                local_lower = base_lower_thresh - 20
                local_upper = base_upper_thresh + 25
                marker_size = base_marker_size + 2
                lv_y, lv_x = 218 + y_offset, 294
                la_y, la_x = 248 + y_offset, 244
                
        elif slice_num < 115:
            if phase == 'diastole':
                local_lower = base_lower_thresh
                local_upper = base_upper_thresh
                marker_size = base_marker_size
                lv_y, lv_x = 225 + y_offset, 300
                la_y, la_x = 260 + y_offset, 250
            else:  
                local_lower = base_lower_thresh - 15
                local_upper = base_upper_thresh + 20
                marker_size = base_marker_size + 1
                lv_y, lv_x = 222 + y_offset, 297
                la_y, la_x = 245 + y_offset, 247
                
        elif slice_num < 120: 
            if phase == 'diastole':
                local_lower = base_lower_thresh
                local_upper = base_upper_thresh
                marker_size = base_marker_size
                lv_y, lv_x = 225 + y_offset, 300
                la_y, la_x = 260 + y_offset, 250
            else: 
                local_lower = base_lower_thresh - 10
                local_upper = base_upper_thresh + 15
                marker_size = base_marker_size
                lv_y, lv_x = 225 + y_offset, 300
                la_y, la_x = 250 + y_offset, 250
                
        else: 
            if phase == 'diastole':
                local_lower = base_lower_thresh - 20
                local_upper = base_upper_thresh + 40
                marker_size = base_marker_size + 3
                lv_y, lv_x = 230 + y_offset, 310
                la_y, la_x = 265 + y_offset, 260
            else:  
                local_lower = base_lower_thresh - 15
                local_upper = base_upper_thresh + 35
                marker_size = base_marker_size - 1  
                lv_y, lv_x = 226 + y_offset, 302
                la_y, la_x = 255 + y_offset, 255
    else:
        local_lower = base_lower_thresh
        local_upper = base_upper_thresh
        marker_size = base_marker_size
        lv_y, lv_x = 225 + y_offset, 300
        la_y, la_x = 260 + y_offset, 250
    
    filt_d = ndi.gaussian_filter(image, filter_sigma)
    mask_d = (filt_d > local_lower) & (filt_d < local_upper)
    
    if phase == 'systole':
        if slice_num is not None and slice_num <= 95:
            clean_d = ndi.binary_opening(mask_d, iterations=1)
            clean_d = ndi.binary_closing(clean_d, iterations=5)  
        elif slice_num is not None and slice_num < 100:
            clean_d = ndi.binary_opening(mask_d, iterations=1)
            clean_d = ndi.binary_closing(clean_d, iterations=4)
        elif slice_num is not None and slice_num == 110:
            clean_d = ndi.binary_opening(mask_d, iterations=1)
            clean_d = ndi.binary_closing(clean_d, iterations=3)
            clean_d = ndi.binary_dilation(clean_d, iterations=1)
            clean_d = ndi.binary_erosion(clean_d, iterations=1)
        elif slice_num is not None and slice_num > 120:
            clean_d = ndi.binary_opening(mask_d, iterations=1)
            clean_d = ndi.binary_closing(clean_d, iterations=2)
        else:
            clean_d = ndi.binary_opening(mask_d, iterations=1)
            clean_d = ndi.binary_closing(clean_d, iterations=2)
    else:
        if slice_num is not None and slice_num > 120:
            clean_d = ndi.binary_opening(mask_d, iterations=1)
            clean_d = ndi.binary_closing(clean_d, iterations=3)
        else:
            clean_d = ndi.binary_opening(mask_d, iterations=2)
            clean_d = ndi.binary_closing(clean_d, iterations=2)
    
    distance = ndi.distance_transform_edt(clean_d)
    
    markers = np.zeros_like(clean_d, dtype=int)
    markers[~clean_d] = 1
    
    if phase == 'systole':
        if slice_num is not None:
            if slice_num <= 95:
                y_radius = marker_size
                x_radius = marker_size
                elliptical_factor = 1.0  
            elif slice_num < 100:
                y_radius = marker_size
                x_radius = marker_size
                elliptical_factor = 1.0  
            elif slice_num < 110:
                y_radius = marker_size
                x_radius = int(marker_size * 0.9)
                elliptical_factor = 0.9  
            elif slice_num < 120:
                y_radius = marker_size
                x_radius = int(marker_size * 0.85)
                elliptical_factor = 0.85 
            else:
                y_radius = marker_size + 1
                x_radius = int(marker_size * 0.8)
                elliptical_factor = 0.8  
        else:
            y_radius = marker_size
            x_radius = int(marker_size * 0.9)
            elliptical_factor = 0.9
    else:
        y_radius = marker_size
        x_radius = marker_size
        elliptical_factor = 1.0
        
    y, x = np.ogrid[-y_radius:y_radius+1, -x_radius:x_radius+1]
    ellipse = (x*x)/(x_radius*x_radius) + (y*y)/(y_radius*y_radius) <= 1
    
    y_min = max(0, lv_y - y_radius)
    y_max = min(image.shape[0], lv_y + y_radius + 1)
    x_min = max(0, lv_x - x_radius)
    x_max = min(image.shape[1], lv_x + x_radius + 1)
    
    e_y_min = max(0, y_radius - lv_y)
    e_y_max = min(2*y_radius+1, y_radius + image.shape[0] - lv_y)
    e_x_min = max(0, x_radius - lv_x)
    e_x_max = min(2*x_radius+1, x_radius + image.shape[1] - lv_x)
    
    ellipse_section = ellipse[e_y_min:e_y_max, e_x_min:e_x_max]
    markers[y_min:y_max, x_min:x_max][ellipse_section] = 2
    
    if phase == 'systole':
        if slice_num is None or slice_num < 110:  
            la_marker_size = 4  
            la_y_min = max(0, la_y-la_marker_size)
            la_y_max = min(image.shape[0], la_y+la_marker_size+1)
            la_x_min = max(0, la_x-la_marker_size)
            la_x_max = min(image.shape[1], la_x+la_marker_size+1)
            markers[la_y_min:la_y_max, la_x_min:la_x_max] = 3
    else:
        if slice_num is None or slice_num < 125:
            la_marker_size = 5
            la_y_min = max(0, la_y-la_marker_size)
            la_y_max = min(image.shape[0], la_y+la_marker_size+1)
            la_x_min = max(0, la_x-la_marker_size)
            la_x_max = min(image.shape[1], la_x+la_marker_size+1)
            markers[la_y_min:la_y_max, la_x_min:la_x_max] = 3
    
    labels = watershed(-distance, markers, mask=clean_d)
    
    lv_mask = labels == 2
    
    if phase == 'systole':
        if slice_num is not None:
            if slice_num <= 95:
                lv_mask = ndi.binary_fill_holes(lv_mask)
                lv_mask = ndi.binary_dilation(lv_mask, iterations=2)
                lv_mask = ndi.binary_erosion(lv_mask, iterations=1)
            elif slice_num < 100:
                lv_mask = ndi.binary_fill_holes(lv_mask)
                lv_mask = ndi.binary_dilation(lv_mask, iterations=1)
                lv_mask = ndi.binary_erosion(lv_mask, iterations=1)
            elif slice_num == 110:
                lv_mask = ndi.binary_fill_holes(lv_mask)
                labeled_regions, num_regions = ndi.label(lv_mask)
                if num_regions > 1:
                    region_sizes = ndi.sum(lv_mask, labeled_regions, range(1, num_regions + 1))
                    largest_region = np.argmax(region_sizes) + 1
                    lv_mask = labeled_regions == largest_region
                lv_mask = ndi.binary_dilation(lv_mask, iterations=1)
                lv_mask = ndi.binary_erosion(lv_mask, iterations=1)
            elif slice_num > 120:
                lv_mask = ndi.binary_fill_holes(lv_mask)
            else:
                labeled_regions, num_regions = ndi.label(lv_mask)
                if num_regions > 1:
                    region_sizes = ndi.sum(lv_mask, labeled_regions, range(1, num_regions + 1))
                    largest_region = np.argmax(region_sizes) + 1
                    lv_mask = labeled_regions == largest_region
                
                lv_mask = ndi.binary_fill_holes(lv_mask)
                lv_mask = ndi.binary_dilation(lv_mask, iterations=1)
                lv_mask = ndi.binary_erosion(lv_mask, iterations=1)
        else:
            lv_mask = ndi.binary_fill_holes(lv_mask)
    else:
        if slice_num is not None and slice_num > 120:
            lv_mask = ndi.binary_fill_holes(lv_mask)
            
            if slice_num > 130:
                lv_mask = ndi.binary_dilation(lv_mask, iterations=1)
        else:
            labeled_regions, num_regions = ndi.label(lv_mask)
            if num_regions > 1:
                region_sizes = ndi.sum(lv_mask, labeled_regions, range(1, num_regions + 1))
                largest_region = np.argmax(region_sizes) + 1
                lv_mask = labeled_regions == largest_region
            
            lv_mask = ndi.binary_fill_holes(lv_mask)
    
    centroids = ndi.center_of_mass(clean_d, labels, range(1, np.max(labels) + 1)) if np.max(labels) > 0 else []
    sizes = ndi.sum(clean_d, labels, range(1, np.max(labels) + 1)) if np.max(labels) > 0 else []
    scores = [1.0 if i == 1 else 0.0 for i in range(np.max(labels))] if np.max(labels) > 0 else []
    
    return lv_mask, scores, centroids, labels


def visualize_candidates(image, labels, scores, centroids):
    """
    Visualize all candidate regions with their scores
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.imshow(image, cmap='gray')
    
    nlabels = len(scores)
    for i in range(nlabels):
        mask = labels == (i + 1)
        if np.sum(mask) > 0:
            score_color = plt.cm.jet(scores[i])
            y, x = centroids[i]
            
            masked_data = np.ma.masked_where(~mask, mask)
            ax.imshow(masked_data, cmap=plt.cm.jet, alpha=0.5, vmin=0, vmax=1)
            
            ax.text(x, y, f"{scores[i]:.2f}", color='white', 
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=1))
    
    ax.set_title("Candidate Regions with Scores")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_segmentation_steps(image, filtered, thresholded, cleaned, final_mask):
    """
    Visualize each step of the segmentation process
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(thresholded, cmap='gray')
    axes[0, 1].set_title('Thresholded Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cleaned, cmap='gray')
    axes[1, 0].set_title('Cleaned Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image, cmap='gray')
    axes[1, 1].imshow(final_mask, alpha=0.3, cmap='Reds')
    axes[1, 1].set_title('LV Segmentation Result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

test_slice_num = 90
test_image = vol_d[test_slice_num]

test_filtered = ndi.gaussian_filter(test_image, 1)

test_thresh = (test_filtered > 250) & (test_filtered < 400)

test_clean = ndi.binary_opening(test_thresh, iterations=2)
test_clean = ndi.binary_closing(test_clean, iterations=3)

lv_mask, scores, centroids, labels = segment_left_ventricle(test_image)

visualize_segmentation_steps(test_image, test_filtered, test_thresh, test_clean, lv_mask)

if len(scores) > 0:
    visualize_candidates(test_image, labels, scores, centroids)

lv_masks_d = []
selected_slices = [90, 95, 105, 115, 125, 135, 140]
segmentation_results = {}
volume_d = np.zeros(end_slice_d - start_slice_d + 1)

for i in range(start_slice_d, end_slice_d + 1):
    im_d = vol_d[i]
    
    lv_mask, scores, centroids, labels = segment_left_ventricle(
        im_d, phase='diastole', slice_num=i, 
        lower_thresh=250, upper_thresh=400
    )
    
    nvoxels_d = np.sum(lv_mask)
    volume_d[i - start_slice_d] = nvoxels_d * dvoxel_d
    
    lv_masks_d.append((i, lv_mask))
    
    if i in selected_slices:
        segmentation_results[i] = {
            'image': im_d,
            'mask': lv_mask,
            'scores': scores,
            'centroids': centroids,
            'labels': labels
        }

total_volume_d = np.sum(volume_d) / 1000
print(f"End Diastolic Volume (EDV): {total_volume_d:.2f} ml")

rows = (len(selected_slices) + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
axes = axes.flatten()

for idx, slice_num in enumerate(selected_slices):
    if slice_num in segmentation_results:
        result = segmentation_results[slice_num]
        
        axes[idx].imshow(result['image'], cmap='gray')
        axes[idx].imshow(result['mask'], alpha=0.3, cmap='Reds')
        axes[idx].set_title(f'Diastolic Slice {slice_num}')
        axes[idx].axis('off')

for idx in range(len(selected_slices), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

vol_s = iio.volread('/kaggle/input/cardiac-ct-dicom-dataset/Systolic1', 'DICOM')

start_slice_s = 95
end_slice_s = 125

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(vol_s[start_slice_s], cmap='gray')
axes[0].set_title(f"Systolic Start Slice ({start_slice_s})")
axes[0].axis('off')

axes[1].imshow(vol_s[end_slice_s], cmap='gray')
axes[1].set_title(f"Systolic End Slice ({end_slice_s})")
axes[1].axis('off')

plt.show()

volume_s = np.zeros(end_slice_s - start_slice_s + 1)

d0_s, d1_s, d2_s = vol_s.meta['sampling']
dvoxel_s = d0_s * d1_s * d2_s 

lv_masks_s = []
selected_s_slices = [95, 100, 105, 110, 115, 120, 125]
systolic_results = {}
volume_s = np.zeros(end_slice_s - start_slice_s + 1)

for i in range(start_slice_s, end_slice_s + 1):
    im_s = vol_s[i]
    
    lv_mask, scores, centroids, labels = segment_left_ventricle(
        im_s, phase='systole', slice_num=i, 
        lower_thresh=250, upper_thresh=400
    )
    
    nvoxels_s = np.sum(lv_mask)
    volume_s[i - start_slice_s] = nvoxels_s * dvoxel_s
    
    lv_masks_s.append((i, lv_mask))
    
    if i in selected_s_slices:
        systolic_results[i] = {
            'image': im_s,
            'mask': lv_mask,
            'scores': scores,
            'centroids': centroids,
            'labels': labels
        }

total_volume_s = np.sum(volume_s) / 1000 
print(f"End Systolic Volume (ESV): {total_volume_s:.2f} ml")

rows = (len(selected_s_slices) + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
axes = axes.flatten()

for idx, slice_num in enumerate(selected_s_slices):
    if slice_num in systolic_results:
        result = systolic_results[slice_num]
        
        axes[idx].imshow(result['image'], cmap='gray')
        axes[idx].imshow(result['mask'], alpha=0.3, cmap='Reds')
        axes[idx].set_title(f'Systolic Slice {slice_num}')
        axes[idx].axis('off')

for idx in range(len(selected_s_slices), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

reference_edv = 93.00  
reference_esv = 33.00  
reference_ef = 64.50   

edv = total_volume_d
esv = total_volume_s
ef_percent = ((edv - esv) / edv) * 100

def calculate_improved_error(calculated_edv, calculated_esv, calculated_ef,
                           reference_edv, reference_esv, reference_ef):
    """
    Calculate improved error metrics for LV volume and EF measurements
    """
    edv_absolute_error = abs(calculated_edv - reference_edv)
    esv_absolute_error = abs(calculated_esv - reference_esv)
    ef_absolute_error = abs(calculated_ef - reference_ef)
    
    edv_percent_error = (edv_absolute_error / reference_edv) * 100
    esv_percent_error = (esv_absolute_error / reference_esv) * 100
    ef_percent_error = (ef_absolute_error / reference_ef) * 100
    
    reference_sv = reference_edv - reference_esv
    calculated_sv = calculated_edv - calculated_esv
    sv_absolute_error = abs(calculated_sv - reference_sv)
    sv_percent_error = (sv_absolute_error / reference_sv) * 100
    
    normalized_volume_error = (edv_absolute_error + esv_absolute_error) / (reference_edv + reference_esv) * 100
    
    rmse_volumes = np.sqrt((edv_percent_error**2 + esv_percent_error**2) / 2)
    

    improved_weighted_error = (0.25 * edv_percent_error + 
                              0.25 * esv_percent_error + 
                              0.4 * ef_percent_error +
                              0.1 * sv_percent_error)
    
    ef_boundaries = [40, 55, 70]
    reference_category = 0
    calculated_category = 0
    
    for i, boundary in enumerate(ef_boundaries):
        if reference_ef >= boundary:
            reference_category = i + 1
        if calculated_ef >= boundary:
            calculated_category = i + 1
    
    category_shift = reference_category != calculated_category
    
    return {
        'edv_absolute_error': edv_absolute_error,
        'esv_absolute_error': esv_absolute_error,
        'ef_absolute_error': ef_absolute_error,
        'edv_percent_error': edv_percent_error,
        'esv_percent_error': esv_percent_error,
        'ef_percent_error': ef_percent_error,
        'sv_percent_error': sv_percent_error,
        'normalized_volume_error': normalized_volume_error,
        'rmse_volumes': rmse_volumes,
        'improved_weighted_error': improved_weighted_error,
        'clinical_category_shift': category_shift
    }

error_metrics = calculate_improved_error(
    calculated_edv=edv,
    calculated_esv=esv,
    calculated_ef=ef_percent,
    reference_edv=reference_edv,
    reference_esv=reference_esv,
    reference_ef=reference_ef
)

print(f"-- Calculated Values --")
print(f"End Diastolic Volume (EDV): {edv:.2f} ml")
print(f"End Systolic Volume (ESV): {esv:.2f} ml")
print(f"Ejection Fraction (EF): {ef_percent:.2f}%")

print(f"\n-- Reference Values --")
print(f"Reference EDV: {reference_edv:.2f} ml")
print(f"Reference ESV: {reference_esv:.2f} ml")
print(f"Reference EF: {reference_ef:.2f}%")

print(f"\n-- Error Analysis --")
print(f"EDV Error: {error_metrics['edv_percent_error']:.2f}%")
print(f"ESV Error: {error_metrics['esv_percent_error']:.2f}%")
print(f"EF Error: {error_metrics['ef_percent_error']:.2f}%")
print(f"Stroke Volume Error: {error_metrics['sv_percent_error']:.2f}%")
print(f"Normalized Volume Error: {error_metrics['normalized_volume_error']:.2f}%")
print(f"RMSE Volumes: {error_metrics['rmse_volumes']:.2f}%")
print(f"Improved Weighted Error: {error_metrics['improved_weighted_error']:.2f}%")

if ef_percent > 70:
    classification = "High Function (>70%)"
elif ef_percent >= 55:
    classification = "Normal Function (55 to 70%)"
elif ef_percent >= 40:
    classification = "Low Function (40 to 55%)"
else:
    classification = "Possible Heart Failure (<40%)"

if reference_ef > 70:
    reference_classification = "High Function (>70%)"
elif reference_ef >= 55:
    reference_classification = "Normal Function (55 to 70%)"
elif reference_ef >= 40:
    reference_classification = "Low Function (40 to 55%)"
else:
    reference_classification = "Possible Heart Failure (<40%)"

print(f"\n-- Clinical Classification --")
print(f"Calculated Classification: {classification}")
print(f"Reference Classification: {reference_classification}")
print(f"Clinical Category Change: {'Yes' if error_metrics['clinical_category_shift'] else 'No'}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

x = np.arange(2)
width = 0.35

axes[0, 0].bar(x - width/2, [edv, esv], width, label='Calculated', color=['#3498db', '#e74c3c'])
axes[0, 0].bar(x + width/2, [reference_edv, reference_esv], width, label='Reference', color=['#85C1E9', '#F1948A'])
axes[0, 0].set_ylabel('Volume (ml)')
axes[0, 0].set_title('LV Volume Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(['EDV', 'ESV'])
axes[0, 0].legend()

for i, v in enumerate([edv, esv]):
    axes[0, 0].text(i - width/2, v + 2, f"{v:.1f}", ha='center')
    
for i, v in enumerate([reference_edv, reference_esv]):
    axes[0, 0].text(i + width/2, v + 2, f"{v:.1f}", ha='center')

axes[0, 1].pie([ef_percent, 100-ef_percent], 
          explode=(0.1, 0), 
          labels=[f'Ejected\n{ef_percent:.1f}%', f'Remaining\n{100-ef_percent:.1f}%'],
          colors=['#e74c3c', '#3498db'],
          autopct='%1.1f%%', 
          shadow=True, 
          startangle=90)
axes[0, 1].set_title(f'Calculated EF: {ef_percent:.1f}%\nReference EF: {reference_ef:.1f}%')
axes[0, 1].axis('equal')

error_labels = ['EDV', 'ESV', 'EF', 'Norm Vol', 'RMSE']
error_values = [error_metrics['edv_percent_error'], 
                error_metrics['esv_percent_error'], 
                error_metrics['ef_percent_error'],
                error_metrics['normalized_volume_error'],
                error_metrics['rmse_volumes']]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
axes[1, 0].bar(error_labels, error_values, color=colors)
axes[1, 0].set_ylabel('Percent Error (%)')
axes[1, 0].set_title('Error Analysis by Metric')
axes[1, 0].axhline(y=3, linestyle='--', color='r', label='Target (<3%)')
axes[1, 0].legend()

error_text = f"""
Error Summary:
- EDV Error: {error_metrics['edv_percent_error']:.2f}%
- ESV Error: {error_metrics['esv_percent_error']:.2f}%
- EF Error: {error_metrics['ef_percent_error']:.2f}%
- Normalized Volume Error: {error_metrics['normalized_volume_error']:.2f}%
- RMSE Volumes: {error_metrics['rmse_volumes']:.2f}%
- Improved Weighted Error: {error_metrics['improved_weighted_error']:.2f}%
- Clinical Impact: {'Category Change' if error_metrics['clinical_category_shift'] else 'No Category Change'}
"""

axes[1, 1].axis('off')
axes[1, 1].text(0.1, 0.5, error_text, fontsize=12, va='center')

plt.tight_layout()
plt.show()


def visualize_3d(volume_masks, title):
    if len(volume_masks) == 0:
        print("No masks available for visualization")
        return
    
    z_max = max([z for z, _ in volume_masks]) + 1
    z_min = min([z for z, _ in volume_masks])
    mask_shape = volume_masks[0][1].shape
    mask_3d = np.zeros((z_max - z_min, mask_shape[0], mask_shape[1]), dtype=bool)
    
    for z, mask in volume_masks:
        if z_min <= z < z_max:
            mask_3d[z - z_min] = mask
    
    z_coords, y_coords, x_coords = np.where(mask_3d)
    z_coords += z_min  
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(z_coords) > 0:
        step = max(1, len(z_coords) // 5000) 
        ax.scatter(x_coords[::step], y_coords[::step], z_coords[::step], 
                   c='r', marker='.', alpha=0.5, s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

visualize_3d(lv_masks_d, '3D Visualization - Left Ventricle (Diastole)')
visualize_3d(lv_masks_s, '3D Visualization - Left Ventricle (Systole)')

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 12))

def add_box(x, y, width, height, text, facecolor='#D9E6F3'):
    rect = plt.Rectangle((x, y), width, height, facecolor=facecolor, edgecolor='black', lw=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', wrap=True, fontsize=10)

def add_arrow(x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=2))


add_box(1, 9, 2, 0.5, "1. Load DICOM Data", facecolor='#C9DAEF')
add_arrow(2, 9, 2, 8.5)

add_box(1, 8, 2, 0.5, "2. Define LV Start/End Slices", facecolor='#C9DAEF')
add_arrow(2, 8, 2, 7.5)

add_box(1, 7, 2, 0.5, "3. Phase-Specific Parameter Selection", facecolor='#AED3F9')
add_arrow(2, 7, 2, 6.5)

add_box(1, 6, 2, 0.5, "4. Position-Aware Parameter Refinement", facecolor='#AED3F9')
add_arrow(2, 6, 2, 5.5)

add_box(1, 5, 2, 0.5, "5. Segmentation Process", facecolor='#AED3F9')

left_branch_x = 0.5
left_branch_width = 1.5
left_center = left_branch_x + left_branch_width/2 

right_branch_x = 2.5
right_branch_width = 1.5
right_center = right_branch_x + right_branch_width/2 

add_arrow(1.25, 5, 1.25, 4.5)  
add_box(left_branch_x, 4.0, left_branch_width, 0.5, "a. Apply Gaussian Filter", facecolor='#E5EFF9')

add_arrow(1.25, 4.0, 1.25, 3.5)  
add_box(left_branch_x, 3.3, left_branch_width, 0.5, "b. Apply Thresholding", facecolor='#E5EFF9')

add_arrow(1.25, 3.3, 1.25, 2.8)  
add_box(left_branch_x, 2.6, left_branch_width, 0.5, "c. Morphological Operations", facecolor='#E5EFF9')

add_arrow(3.25, 5, 3.25, 4.5)  
add_box(right_branch_x, 4.0, right_branch_width, 0.5, "d. Create Watershed Markers", facecolor='#E5EFF9')

add_arrow(3.25, 4.0, 3.25, 3.5)  
add_box(right_branch_x, 3.3, right_branch_width, 0.5, "e. Apply Watershed Segmentation", facecolor='#E5EFF9')

add_arrow(3.25, 3.3, 3.25, 2.8)  
add_box(right_branch_x, 2.6, right_branch_width, 0.5, "f. Phase-Specific Post-Processing", facecolor='#E5EFF9')


merge_point = (2.25, 2.2)
add_arrow(1.25, 2.6, merge_point[0], merge_point[1])  
add_arrow(3.25, 2.6, merge_point[0], merge_point[1])  

add_arrow(merge_point[0], merge_point[1], 2, 1.5)


add_box(1, 1, 2, 0.5, "6. Calculate Volume for All Slices", facecolor='#C9DAEF')
add_arrow(2, 1, 2, 0.5)

add_box(1, 0, 2, 0.5, "7. Calculate Ejection Fraction & Classify", facecolor='#C9DAEF')


ax.set_xlim(0, 4.5)
ax.set_ylim(-0.5, 10.5)
ax.axis('off')
ax.set_title('Phase and Position-Aware Left Ventricle Segmentation Algorithm', fontsize=14)

plt.tight_layout()
plt.show()
