#!/usr/bin/env python3
"""
PyQt5 GUI for Manual Threshold Validation Results Collection
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QTextEdit, QComboBox, QSpinBox, QGroupBox, 
                             QFileDialog, QMessageBox, QProgressBar, QScrollArea)
from PyQt5.QtCore import Qt, QSize, QUrl
from PyQt5.QtGui import QFont, QPixmap, QImage, QDesktopServices, QColor
import subprocess
import platform
import numpy as np
from tifffile import imread

class ValidationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.output_dir = Path("/Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning/Test thresholding")
        self.image_dirs = []
        self.current_image_idx = 0
        self.results = {
            "validation_date": datetime.now().strftime("%Y-%m-%d"),
            "validator_name": "",
            "notes": "",
            "images": []
        }
        self.init_ui()
        self.load_images()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Threshold Validation Tool")
        self.setGeometry(50, 50, 1400, 1000)
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 15px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 18px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                font-size: 17px;
                font-weight: bold;
                color: #000000;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#navButton {
                background-color: #2196F3;
            }
            QPushButton#navButton:hover {
                background-color: #0b7dda;
            }
            QPushButton#saveButton {
                background-color: #FF9800;
            }
            QPushButton#saveButton:hover {
                background-color: #e68900;
            }
            QLineEdit, QSpinBox, QComboBox, QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                color: #000000;
            }
            QLineEdit {
                color: #000000;
            }
            QSpinBox {
                color: #000000;
            }
            QSpinBox::up-button {
                background-color: #e0e0e0;
                border: 1px solid #999;
                border-radius: 2px;
                width: 20px;
                height: 15px;
            }
            QSpinBox::up-button:hover {
                background-color: #d0d0d0;
            }
            QSpinBox::down-button {
                background-color: #e0e0e0;
                border: 1px solid #999;
                border-radius: 2px;
                width: 20px;
                height: 15px;
            }
            QSpinBox::down-button:hover {
                background-color: #d0d0d0;
            }
            QSpinBox::up-arrow {
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 7px solid #000000;
            }
            QSpinBox::up-arrow:disabled {
                border-bottom: 7px solid #999999;
            }
            QSpinBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #000000;
            }
            QSpinBox::down-arrow:disabled {
                border-top: 7px solid #999999;
            }
            QComboBox {
                color: #000000;
            }
            QComboBox QAbstractItemView {
                color: #000000;
                background-color: white;
            }
            QTextEdit {
                color: #000000;
            }
            QLabel#titleLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                padding: 5px;
            }
            QLabel {
                font-size: 11px;
            }
            QGroupBox {
                font-size: 12px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Threshold Validation Tool")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title.setMaximumHeight(40)
        main_layout.addWidget(title)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p% (%v/%m)")
        main_layout.addWidget(self.progress_bar)
        
        # Content widget (no scrolling - make everything fit)
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setSpacing(8)
        main_layout.addWidget(content_widget)
        
        # Store name and notes in results but don't show UI for them
        self.name_input = QLineEdit()
        self.name_input.setVisible(False)
        self.notes_input = QTextEdit()
        self.notes_input.setVisible(False)
        
        # Image info section
        self.image_info_group = QGroupBox("Current Image")
        image_info_layout = QVBoxLayout()
        self.image_name_label = QLabel("No image loaded")
        self.image_name_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.image_name_label.setMaximumHeight(30)
        image_info_layout.addWidget(self.image_name_label)
        
        help_label = QLabel("📋 Instructions: Open ImageJ and review the overlay images at the paths below.\n"
                           "Compare different thresholds (0-16) to find the best one for this image.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("background-color: #e3f2fd; padding: 5px; border-radius: 4px; color: #1976d2; font-size: 10px;")
        help_label.setMaximumHeight(40)
        image_info_layout.addWidget(help_label)
        
        # File paths with clickable links
        paths_container = QVBoxLayout()
        paths_container.setSpacing(2)
        
        # 2D path
        path_2d_label = QLabel("")
        path_2d_label.setWordWrap(True)
        path_2d_label.setStyleSheet("color: #0066cc; font-size: 10px; text-decoration: underline;")
        path_2d_label.setCursor(Qt.PointingHandCursor)
        path_2d_label.mousePressEvent = lambda e, label=path_2d_label: self.open_path_2d()
        self.path_2d_label = path_2d_label
        paths_container.addWidget(path_2d_label)
        
        # 3D path
        path_3d_label = QLabel("")
        path_3d_label.setWordWrap(True)
        path_3d_label.setStyleSheet("color: #0066cc; font-size: 10px; text-decoration: underline;")
        path_3d_label.setCursor(Qt.PointingHandCursor)
        path_3d_label.mousePressEvent = lambda e, label=path_3d_label: self.open_path_3d()
        self.path_3d_label = path_3d_label
        paths_container.addWidget(path_3d_label)
        
        # Initialize paths
        self.path_2d = ""
        self.path_3d = ""
        
        image_info_layout.addLayout(paths_container)
        
        # Spot counts display
        self.spot_counts_label = QLabel("")
        self.spot_counts_label.setWordWrap(True)
        self.spot_counts_label.setStyleSheet("font-family: monospace; font-size: 9px;")
        self.spot_counts_label.setMaximumHeight(60)
        image_info_layout.addWidget(self.spot_counts_label)
        
        self.image_info_group.setLayout(image_info_layout)
        self.content_layout.addWidget(self.image_info_group)
        
        # Validation inputs section
        validation_group = QGroupBox("Validation Inputs - Fill these based on your ImageJ review")
        validation_layout = QVBoxLayout()
        validation_layout.setSpacing(6)
        
        # Best threshold
        thresh_container = QVBoxLayout()
        thresh_instruction = QLabel("1. Best Threshold: After reviewing overlays in ImageJ, select the threshold index (0-16) that gives the best spot detection.\n   Index 0 = lowest threshold (most spots), Index 16 = highest threshold (fewest spots).\n   Actual threshold values are calculated dynamically from detection results.")
        thresh_instruction.setWordWrap(True)
        thresh_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        thresh_container.addWidget(thresh_instruction)
        thresh_layout = QHBoxLayout()
        thresh_layout.setAlignment(Qt.AlignLeft)
        thresh_layout.setContentsMargins(0, 0, 0, 0)
        self.best_threshold = QSpinBox()
        self.best_threshold.setRange(0, 16)
        self.best_threshold.setValue(3)
        self.best_threshold.setMinimumWidth(80)
        self.best_threshold.valueChanged.connect(self.update_threshold_display)
        thresh_layout.addWidget(self.best_threshold)
        self.threshold_value_label = QLabel("(threshold: calculating...)")
        self.threshold_value_label.setStyleSheet("color: #666; font-size: 10px;")
        thresh_layout.addWidget(self.threshold_value_label)
        thresh_layout.addStretch()
        thresh_container.addLayout(thresh_layout)
        validation_layout.addLayout(thresh_container)
        
        # Initialize threshold values
        self.threshold_min = 0
        self.threshold_max = 16
        self.threshold_values = list(range(17))
        
        # Alternative thresholds
        alt_container = QVBoxLayout()
        alt_instruction = QLabel("2. Alternative Thresholds: Other threshold values that also work well (optional).")
        alt_instruction.setWordWrap(True)
        alt_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        alt_container.addWidget(alt_instruction)
        alt_layout = QHBoxLayout()
        alt_layout.setAlignment(Qt.AlignLeft)
        alt_layout.setContentsMargins(0, 0, 0, 0)
        self.alt_thresholds = QLineEdit()
        self.alt_thresholds.setPlaceholderText("e.g., 2, 4 (or leave empty if none)")
        self.alt_thresholds.setMinimumWidth(250)
        alt_layout.addWidget(self.alt_thresholds)
        alt_layout.addStretch()
        alt_container.addLayout(alt_layout)
        validation_layout.addLayout(alt_container)
        
        # Spot count range
        range_container = QVBoxLayout()
        range_instruction = QLabel("3. Reasonable Spot Count Range: Based on your review, what's a reasonable range of spot counts for this image?")
        range_instruction.setWordWrap(True)
        range_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        range_container.addWidget(range_instruction)
        range_layout = QHBoxLayout()
        range_layout.setAlignment(Qt.AlignLeft)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(QLabel("Minimum:"))
        self.min_spots = QSpinBox()
        self.min_spots.setRange(0, 1000000)
        self.min_spots.setMinimumWidth(100)
        range_layout.addWidget(self.min_spots)
        range_layout.addWidget(QLabel("Maximum:"))
        self.max_spots = QSpinBox()
        self.max_spots.setRange(0, 1000000)
        self.max_spots.setMinimumWidth(100)
        range_layout.addWidget(self.max_spots)
        range_layout.addStretch()
        range_container.addLayout(range_layout)
        validation_layout.addLayout(range_container)
        
        # Quality rating
        quality_container = QVBoxLayout()
        quality_instruction = QLabel("4. Overall Quality Rating: How good is the spot detection at the best threshold?\n   Poor = 70% or less, Fair = 80% or more, Good = 90% or more, Excellent = 95% or more")
        quality_instruction.setWordWrap(True)
        quality_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        quality_container.addWidget(quality_instruction)
        quality_layout = QHBoxLayout()
        quality_layout.setAlignment(Qt.AlignLeft)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Poor (≤70%)", "Fair (≥80%)", "Good (≥90%)", "Excellent (≥95%)"])
        self.quality_combo.setCurrentIndex(2)  # Default to "Good"
        self.quality_combo.setMinimumWidth(150)
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        quality_container.addLayout(quality_layout)
        validation_layout.addLayout(quality_container)
        
        # Observations
        obs_container = QVBoxLayout()
        obs_instruction = QLabel("5. Observations: Describe what you observed when reviewing the overlays in ImageJ.")
        obs_instruction.setWordWrap(True)
        obs_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        obs_container.addWidget(obs_instruction)
        self.observations = QTextEdit()
        self.observations.setMaximumHeight(50)
        self.observations.setPlaceholderText("Example: 'Threshold 3 shows good balance - captures most real spots without too many false positives'")
        obs_container.addWidget(self.observations)
        validation_layout.addLayout(obs_container)
        
        # Issues
        issues_container = QVBoxLayout()
        issues_instruction = QLabel("6. Issues: Note any problems you noticed (or write 'None' if no issues).")
        issues_instruction.setWordWrap(True)
        issues_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        issues_container.addWidget(issues_instruction)
        self.issues = QTextEdit()
        self.issues.setMaximumHeight(40)
        self.issues.setPlaceholderText("Example: 'Some background noise at lower thresholds' or 'None'")
        issues_container.addWidget(self.issues)
        validation_layout.addLayout(issues_container)
        
        # Recommended for training
        rec_container = QVBoxLayout()
        rec_instruction = QLabel("7. Recommended for Training: Should this image be used for training the ML model?")
        rec_instruction.setWordWrap(True)
        rec_instruction.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; font-size: 11px;")
        rec_container.addWidget(rec_instruction)
        rec_layout = QHBoxLayout()
        rec_layout.setAlignment(Qt.AlignLeft)
        rec_layout.setContentsMargins(0, 0, 0, 0)
        self.recommended = QComboBox()
        self.recommended.addItems(["Yes", "No"])
        self.recommended.setCurrentIndex(0)  # Default to "Yes"
        self.recommended.setMinimumWidth(80)
        rec_layout.addWidget(self.recommended)
        rec_layout.addStretch()
        rec_container.addLayout(rec_layout)
        validation_layout.addLayout(rec_container)
        
        validation_group.setLayout(validation_layout)
        self.content_layout.addWidget(validation_group)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← Previous")
        self.prev_button.setObjectName("navButton")
        self.prev_button.clicked.connect(self.prev_image)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next →")
        self.next_button.setObjectName("navButton")
        self.next_button.clicked.connect(self.next_image)
        nav_layout.addWidget(self.next_button)
        
        nav_layout.addStretch()
        
        self.save_button = QPushButton("💾 Save Results")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.save_results)
        nav_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("📂 Load Results")
        self.load_button.clicked.connect(self.load_results)
        nav_layout.addWidget(self.load_button)
        
        nav_group.setLayout(nav_layout)
        self.content_layout.addWidget(nav_group)
        
        self.content_layout.addStretch()
        
    def load_images(self):
        """Load list of images from output directory"""
        if not self.output_dir.exists():
            QMessageBox.warning(self, "Error", f"Output directory not found: {self.output_dir}")
            return
        
        self.image_dirs = sorted([d for d in self.output_dir.iterdir() 
                                 if d.is_dir() and not d.name.startswith('.')])
        
        if len(self.image_dirs) == 0:
            QMessageBox.warning(self, "Error", f"No image directories found in {self.output_dir}")
            return
        
        self.progress_bar.setMaximum(len(self.image_dirs))
        self.progress_bar.setValue(0)
        self.load_current_image()
        
    def load_current_image(self):
        """Load current image data"""
        if self.current_image_idx >= len(self.image_dirs):
            return
        
        img_dir = self.image_dirs[self.current_image_idx]
        image_name = img_dir.name
        
        # Update UI
        self.image_name_label.setText(f"Image {self.current_image_idx + 1}/{len(self.image_dirs)}: {image_name}")
        
        # Set clickable file paths
        path_2d = str(img_dir)
        path_3d = str(img_dir / '3D_overlays')
        self.path_2d = path_2d
        self.path_3d = path_3d
        
        self.path_2d_label.setText(f"📁 2D: {path_2d}")
        self.path_3d_label.setText(f"📁 3D: {path_3d}")
        
        # Load spot counts from summary and determine threshold range
        summary_file = img_dir / f"{image_name}_565_results_summary.json"
        spot_counts_text = ""
        self.threshold_min = 0
        self.threshold_max = 16
        self.threshold_values = list(range(17))  # Default 0-16
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                    # Find threshold range
                    results = summary.get('results', {})
                    if results:
                        # Convert threshold keys to floats (they might be strings)
                        thresholds = []
                        for k, v in results.items():
                            try:
                                thresh_val = float(k)
                                n_spots = v.get('n_spots', 0)
                                thresholds.append((thresh_val, n_spots))
                            except:
                                continue
                        
                        if thresholds:
                            # Sort by threshold value
                            thresholds.sort(key=lambda x: x[0])
                            
                            # Find min threshold (lowest value, usually gives most spots)
                            self.threshold_min = thresholds[0][0]
                            
                            # Find max threshold (highest value that gives 0 spots, or highest available)
                            max_thresh_with_spots = self.threshold_min
                            for thresh_val, n_spots in thresholds:
                                if n_spots > 0:
                                    max_thresh_with_spots = thresh_val
                            
                            # Find the threshold that gives 0 spots (or use the highest available)
                            self.threshold_max = thresholds[-1][0]
                            for thresh_val, n_spots in reversed(thresholds):
                                if n_spots == 0:
                                    self.threshold_max = thresh_val
                                    break
                            
                            # Create gradient of 16 thresholds
                            threshold_range = self.threshold_max - self.threshold_min
                            step_size = threshold_range / 16.0
                            self.threshold_values = [
                                round(self.threshold_min + i * step_size, 2) 
                                for i in range(17)
                            ]
                            
                            # Update spinbox range
                            self.best_threshold.setRange(0, 16)
                            
                            spot_counts_text = f"Threshold range: {self.threshold_min:.2f} to {self.threshold_max:.2f}\n"
                            spot_counts_text += "Spot counts:\n"
                            for i, (thresh_val, n_spots) in enumerate(thresholds[:10]):
                                spot_counts_text += f"  Threshold {thresh_val:.2f}: {n_spots:,} spots\n"
            except Exception as e:
                spot_counts_text = f"Error loading summary: {e}"
        
        self.spot_counts_label.setText(spot_counts_text)
        
        # Load existing validation data if available
        existing_data = self.get_existing_data(image_name)
        if existing_data:
            # Support both old format (best_threshold as index) and new format
            best_thresh = existing_data.get('best_threshold_index', existing_data.get('best_threshold', 3))
            if isinstance(best_thresh, (int, float)):
                # If it's a threshold value, find closest index
                if best_thresh not in range(17):
                    # Find closest index
                    best_thresh = min(range(len(self.threshold_values)), 
                                    key=lambda i: abs(self.threshold_values[i] - best_thresh))
                self.best_threshold.setValue(int(best_thresh))
            else:
                self.best_threshold.setValue(3)
            alt_thresh = existing_data.get('alternative_thresholds', [])
            self.alt_thresholds.setText(', '.join(map(str, alt_thresh)))
            self.min_spots.setValue(existing_data.get('spot_count_range', {}).get('min_reasonable', 0))
            self.max_spots.setValue(existing_data.get('spot_count_range', {}).get('max_reasonable', 0))
            quality = existing_data.get('quality_rating', 'good')
            quality_idx = ['poor', 'fair', 'good', 'excellent'].index(quality.lower()) if quality.lower() in ['poor', 'fair', 'good', 'excellent'] else 2
            self.quality_combo.setCurrentIndex(quality_idx)
            self.observations.setPlainText(existing_data.get('observations', ''))
            self.issues.setPlainText(existing_data.get('issues', ''))
            self.recommended.setCurrentIndex(0 if existing_data.get('recommended_for_training', True) else 1)
        else:
            # Reset to defaults
            self.best_threshold.setValue(3)
            self.alt_thresholds.clear()
            self.min_spots.setValue(0)
            self.max_spots.setValue(0)
            self.quality_combo.setCurrentIndex(2)
            self.observations.clear()
            self.issues.clear()
            self.recommended.setCurrentIndex(0)
        
        # Update threshold display
        self.update_threshold_display()
        
        # Update progress
        self.progress_bar.setValue(self.current_image_idx + 1)
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_image_idx > 0)
        self.next_button.setEnabled(self.current_image_idx < len(self.image_dirs) - 1)
        
    def update_threshold_display(self):
        """Update the threshold value label when spinbox changes"""
        index = self.best_threshold.value()
        if index < len(self.threshold_values):
            actual_value = self.threshold_values[index]
            self.threshold_value_label.setText(f"(threshold: {actual_value:.2f})")
        else:
            self.threshold_value_label.setText("(threshold: N/A)")
    
    def open_path_2d(self):
        """Open 2D path in Finder"""
        self.open_in_finder(self.path_2d)
    
    def open_path_3d(self):
        """Open 3D path in Finder"""
        self.open_in_finder(self.path_3d)
    
    def open_in_finder(self, path):
        """Open a file path in Finder (macOS) or file browser"""
        if not path:
            return
        
        path_obj = Path(path)
        if not path_obj.exists():
            QMessageBox.warning(self, "Path Not Found", f"The path does not exist:\n{path}")
            return
        
        try:
            if platform.system() == "Darwin":  # macOS
                # Use 'open -R' to reveal in Finder and select the file/folder
                subprocess.run(["open", "-R", str(path_obj)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", "/select,", str(path_obj)])
            else:  # Linux
                subprocess.run(["xdg-open", str(path_obj.parent)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open path:\n{path}\n\nError: {e}")
    
    def get_existing_data(self, image_name):
        """Get existing validation data for an image"""
        for img_data in self.results['images']:
            if img_data['image_name'] == image_name:
                return img_data
        return None
    
    def save_current_data(self):
        """Save current image's validation data"""
        image_name = self.image_dirs[self.current_image_idx].name
        
        # Parse alternative thresholds
        alt_text = self.alt_thresholds.text().strip()
        if alt_text:
            try:
                alt_thresh = [int(x.strip()) for x in alt_text.split(',') if x.strip()]
            except:
                alt_thresh = []
        else:
            alt_thresh = []
        
        # Get actual threshold value from index
        threshold_index = self.best_threshold.value()
        actual_threshold = self.threshold_values[threshold_index] if threshold_index < len(self.threshold_values) else threshold_index
        
        data = {
            "image_name": image_name,
            "best_threshold_index": threshold_index,
            "best_threshold_value": float(actual_threshold),
            "alternative_thresholds": alt_thresh,
            "spot_count_range": {
                "min_reasonable": self.min_spots.value(),
                "max_reasonable": self.max_spots.value()
            },
            "quality_rating": self.quality_combo.currentText().split()[0].lower(),  # Extract "Poor", "Fair", etc. from "Poor (≤70%)"
            "observations": self.observations.toPlainText(),
            "issues": self.issues.toPlainText(),
            "recommended_for_training": self.recommended.currentText() == "Yes"
        }
        
        # Update or add to results
        existing_idx = None
        for i, img_data in enumerate(self.results['images']):
            if img_data['image_name'] == image_name:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.results['images'][existing_idx] = data
        else:
            self.results['images'].append(data)
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_image_idx > 0:
            self.save_current_data()
            self.current_image_idx -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.current_image_idx < len(self.image_dirs) - 1:
            self.save_current_data()
            self.current_image_idx += 1
            self.load_current_image()
    
    def update_validator_name(self, text):
        """Update validator name"""
        self.results['validator_name'] = text
    
    def update_notes(self):
        """Update general notes"""
        self.results['notes'] = self.notes_input.toPlainText()
    
    def save_results(self):
        """Save all results to JSON file"""
        self.save_current_data()
        
        # Update metadata
        self.results['validation_date'] = datetime.now().strftime("%Y-%m-%d")
        # Name and notes are optional, keep existing or set empty
        if 'validator_name' not in self.results:
            self.results['validator_name'] = ""
        if 'notes' not in self.results:
            self.results['notes'] = ""
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Validation Results", 
            "manual_validation_results.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                QMessageBox.information(self, "Success", 
                    f"Results saved to:\n{file_path}\n\n"
                    f"Validated {len(self.results['images'])} images")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def load_results(self):
        """Load existing results from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Validation Results",
            "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.results = json.load(f)
                
                # Update UI (name and notes are stored but not shown in UI)
                if 'validator_name' in self.results:
                    self.results['validator_name'] = self.results.get('validator_name', '')
                if 'notes' in self.results:
                    self.results['notes'] = self.results.get('notes', '')
                
                # Reload current image to show loaded data
                self.load_current_image()
                
                QMessageBox.information(self, "Success", 
                    f"Loaded results from:\n{file_path}\n\n"
                    f"{len(self.results.get('images', []))} images in file")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.save_current_data()
        reply = QMessageBox.question(self, "Save Before Exit?",
            "Do you want to save your validation results before exiting?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        
        if reply == QMessageBox.Yes:
            self.save_results()
            event.accept()
        elif reply == QMessageBox.Cancel:
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = ValidationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

