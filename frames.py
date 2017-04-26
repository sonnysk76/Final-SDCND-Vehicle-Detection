def process_frame(frame):
    xy_window = (64, 64)
    windows = slide_window(frame, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(frame, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    self.heatmap = add_heat(heatmap, hot_windows)
    xy_window = (128, 128)
    windows = slide_window(frame, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(frame, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    self.heatmap = add_heat(heatmap, hot_windows)
    xy_window = (256, 256)
    windows = slide_window(frame, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(frame, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    self.heatmap = add_heat(heatmap, hot_windows)

    # example using heatmap
    new_frame_factor = 0.3
    self.heatmap = new_frame_factor * frame + (1 - new_frame_factor) * self.heatmap
    self.heatmap = apply_threshold(heatmap, threshold)

    # example using averaged frame
    nb_frames_avg = 15
    self.frames.append(frame)
    avg_frame = np.mean(np.array(self.frames)[-nb_frames_avg], axis=-1)
    labels = label(avg_frame)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    return draw_img