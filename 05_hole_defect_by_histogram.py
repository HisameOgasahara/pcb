import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_hole_histogram(image_path, roi_list):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load the image.")
        return
    
    # Analyze each ROI
    results = []  # Store detection results
    for idx, roi in enumerate(roi_list):
        print(f"\n=== Analysis for ROI {idx} ===")
        
        # Calculate ROI bounding box
        x_min = min(p[0] for p in roi)
        y_min = min(p[1] for p in roi)
        x_max = max(p[0] for p in roi)
        y_max = max(p[1] for p in roi)
        
        # Extract ROI and convert to grayscale
        roi_color = image[y_min:y_max, x_min:x_max].copy()
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # Detect circles
        circles = cv2.HoughCircles(
            roi_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        if circles is None:
            print(f"No circles found in ROI {idx}")
            continue
            
        # Find the largest circle
        circles = np.uint16(np.around(circles))
        largest_circle = circles[0][0]
        
        # Circle center and radius
        center_x, center_y, radius = largest_circle
        
        # Create circular mask
        mask = np.zeros_like(roi_gray)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Extract circle pixels
        circle_pixels = roi_gray[mask == 255]
        
        # Calculate statistics
        stats_dict = {
            "Mean": np.mean(circle_pixels),
            "Median": np.median(circle_pixels),
            "Mode": stats.mode(circle_pixels)[0],
            "Std Dev": np.std(circle_pixels),
            "Variance": np.var(circle_pixels),
            "Min": np.min(circle_pixels),
            "Max": np.max(circle_pixels),
            "Q1": np.percentile(circle_pixels, 25),
            "Q3": np.percentile(circle_pixels, 75),
            "Skewness": stats.skew(circle_pixels),
            "Kurtosis": stats.kurtosis(circle_pixels),
            "Dark pixels ratio (<50)": np.sum(circle_pixels < 50) / len(circle_pixels),
            "Bright pixels ratio (>150)": np.sum(circle_pixels > 150) / len(circle_pixels)
        }

        # Analyze histogram to find multiple peaks
        hist, bin_edges = np.histogram(circle_pixels, bins=50, range=(0, 255))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks in histogram
        peak_indices = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(circle_pixels) * 0.05:
                peak_indices.append(i)
        
        # Get peak values and heights
        peaks = [(bin_centers[i], hist[i]) for i in peak_indices]
        
        # Find the main peak (closest to mode)
        mode_bin_index = np.argmin(np.abs(bin_centers - stats_dict["Mode"]))
        mode_peak_height = hist[mode_bin_index]
        
        # Check for significant secondary peaks
        has_significant_secondary_peak = False
        secondary_peak_info = ""
        
        for peak_val, peak_height in peaks:
            # Ignore peaks close to the mode
            if abs(peak_val - stats_dict["Mode"]) < 10:
                continue
                
            # If there's a peak with height at least 60% of the mode peak
            if peak_height > mode_peak_height * 0.6:
                has_significant_secondary_peak = True
                secondary_peak_info = f"Secondary peak at {peak_val:.1f} with height {peak_height}"
                break
        
        # Store the multi-peak information in stats
        stats_dict["Has Secondary Peak"] = has_significant_secondary_peak
        stats_dict["Secondary Peak Info"] = secondary_peak_info

        # Defect detection logic
        is_defective = False
        defect_reasons = []

        # Check 1: Mode value should be between 10-17
        if not (10 <= stats_dict["Mode"] <= 17):
            is_defective = True
            defect_reasons.append("Mode value out of normal range (10-17)")

        # Check 2: Check for significant secondary peaks
        if has_significant_secondary_peak:
            is_defective = True
            defect_reasons.append(f"Multiple peaks detected: {secondary_peak_info}")

        # Store results
        results.append({
            'roi_idx': idx,
            'is_defective': is_defective,
            'reasons': defect_reasons,
            'stats': stats_dict
        })
        
        # Print detection results
        print("\nDefect Detection Results:")
        print(f"ROI {idx}: {'DEFECTIVE' if is_defective else 'NORMAL'}")
        if is_defective:
            print("Reasons:", ", ".join(defect_reasons))
        
        # Print statistics
        print("\nPixel Statistics:")
        for stat_name, stat_value in stats_dict.items():
            if isinstance(stat_value, (int, float)):
                print(f"{stat_name}: {stat_value:.2f}")
            else:
                print(f"{stat_name}: {stat_value}")
        
        # Visualization with defect status
        plt.figure(figsize=(12, 4))
        
        # Original image with detected circle
        plt.subplot(1, 2, 1)
        title = f'ROI {idx} - {"DEFECTIVE" if is_defective else "NORMAL"}'
        if is_defective:
            title += f'\nReasons: {", ".join(defect_reasons)}'
        plt.title(title, color='red' if is_defective else 'green', fontsize=10)
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        circle_color = (255, 0, 0) if is_defective else (0, 255, 0)
        cv2.circle(roi_rgb, (center_x, center_y), radius, circle_color, 2)
        plt.imshow(roi_rgb)
        plt.axis('off')
        
        # Histogram with peaks highlighted
        plt.subplot(1, 2, 2)
        hist_title = f'ROI {idx} - Pixel Value Distribution\n'
        hist_title += f'Mode: {stats_dict["Mode"]:.1f}, Mean: {stats_dict["Mean"]:.1f}, '
        hist_title += f'Median: {stats_dict["Median"]:.1f}, StdDev: {stats_dict["Std Dev"]:.1f}'
        if has_significant_secondary_peak:
            hist_title += "\nMultiple peaks detected!"
        plt.title(hist_title, fontsize=10)
        
        # Plot histogram
        n, bins, patches = plt.hist(circle_pixels, bins=50, range=(0, 255), color='blue', alpha=0.7)
        
        # Highlight peaks
        for peak_val, peak_height in peaks:
            bin_idx = np.argmin(np.abs(bin_centers - peak_val))
            if abs(peak_val - stats_dict["Mode"]) < 10:
                # Main peak
                patches[bin_idx].set_facecolor('green')
            else:
                # Secondary peak
                patches[bin_idx].set_facecolor('red')
        
        plt.axvline(x=150, color='r', linestyle='--', label='Threshold (150)')
        plt.axvline(x=stats_dict["Mean"], color='g', linestyle='--', label='Mean')
        plt.axvline(x=stats_dict["Median"], color='y', linestyle='--', label='Median')
        plt.axvline(x=stats_dict["Mode"], color='purple', linestyle='--', label='Mode')
        plt.xlabel('Pixel Value')
        plt.ylabel('Pixel Count')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return results

if __name__ == "__main__":
    # Set image path
    image_path = r"C:\Users\hiirgi\Downloads\pcb_eng\converted_images\12th_L50_1.png"
    
    # Define ROI list
    roi_list = [
        [(101, 487), (276, 487), (276, 616), (101, 616)],  # First ROI (rectangle)
        [(101, 638), (276, 638), (276, 742), (101, 742)],  # Second ROI (trapezoid)
        [(101, 781), (284, 781), (284, 903), (101, 903)],  # Third ROI (angled rectangle)
        [(2303, 490), (2488, 490), (2488, 609), (2303, 609)],
        [(2303, 633), (2488, 633), (2488, 762), (2303, 762)],
        [(2308, 776), (2488, 776), (2488, 890), (2308, 890)],
    ]
    
    # Run histogram analysis
    analyze_hole_histogram(image_path, roi_list)
