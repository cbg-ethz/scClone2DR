#### Anne Bertolini
#### March 2025

#### Helper functions to get heat maps from the modelling results of the drSCS project

# with attrs list and gene_info table, replace the gene_ids where possible
# give out vector that is replacing "attrs$dim4_dimensions"
replace_gene_ids_with_symbols <- function(attrs, gene_info) {
  attrs$dim4_dimensions %>%
    as.data.frame() %>%
    dplyr::rename(., gene_id = `.`) %>%
    left_join(gene_info, by = "gene_id") %>%
    mutate(new_name = ifelse(nzchar(gene_name), gene_name, gene_id)) %>% # nzchar TRUE if non-empty string, non-zero character, probably
    pull(new_name)  # Extract the column as a vector
}

# with attrs list and gene_info table, replace the gene_ids where possible
# give out vector that is replacing "attrs$dim2_dimensions"
replace_gene_ids_with_symbols_beta <- function(attrs, gene_info) {
  attrs$dim2_dimensions %>%
    as.data.frame() %>%
    dplyr::rename(., gene_id = `.`) %>%
    left_join(gene_info, by = "gene_id") %>%
    mutate(new_name = ifelse(nzchar(gene_name), gene_name, gene_id)) %>% # nzchar TRUE if non-empty string, non-zero character, probably
    pull(new_name)  # Extract the column as a vector
}

# with attrs list and gene_info table, replace the gene_ids where possible
# give out vector that is replacing "attrs$dim4_dimensions" (LOR)
replace_gene_ids_with_symbols_lor <- function(attrs, gene_info) {
  attrs$dim4_dimensions %>%
    as.data.frame() %>%
    rename(., gene_id = `.`) %>%
    left_join(gene_info, by = "gene_id") %>%
    mutate(new_name = ifelse(nzchar(gene_name), gene_name, gene_id)) %>% # nzchar TRUE if non-empty string, non-zero character, probably
    pull(new_name)  # Extract the column as a vector
}

# variance-specific -------------------------------------------------------


# get variance per feature/drug combination
# single data points of the input are clones/clusters of all samples
# variance(all clones)
get_variance_per_feature <- function(data, drug, feature, attrs) {
  my_data <- as.data.frame(data[drug, , , feature])
  rownames(my_data) <- attrs$dim2_subclones
  colnames(my_data) <- attrs$dim3_samples
  my_long_data <- pivot_longer(my_data,
                               cols = everything(),
                               names_to = "sample",
                               values_to = "local_importance")
  # filter out entries where the sample did not have value for the respective
  # cluster ID
  filter_mask_rows <- is.na(my_long_data$local_importance)
  my_long_data <- my_long_data[!filter_mask_rows, ]
  # variance for drug/feature combination
  my_var <- var(my_long_data$local_importance)
  my_var
}

# reformat the variance table and write it into table
export_variance_in_table <- function(md, my_version, cancer_type) {
  md_export <- md %>%
    rownames_to_column("drug") %>%
    select(drug, everything())

  filename <- paste0(outdirs[[my_version]],
                     cancer_type,
                     "_",
                     my_version,
                     "__drug_feature_variance_local_importance.txt")
  write.table(x = md_export,
              file = filename,
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = T)
}


# mean-specific -----------------------------------------------------------


# get mean of local importance value per feature/drug combination
# single data points of the input are clones/clusters of all samples
get_mean_per_feature <- function(data, drug, feature, attrs) {
  my_data <- as.data.frame(data[drug, , , feature])
  rownames(my_data) <- attrs$dim2_subclones
  colnames(my_data) <- attrs$dim3_samples
  my_long_data <- pivot_longer(my_data,
                               cols = everything(),
                               names_to = "sample",
                               values_to = "local_importance")
  # filter out entries where the sample did not have value for the respective
  # cluster ID
  filter_mask_rows <- is.na(my_long_data$local_importance)
  my_long_data <- my_long_data[!filter_mask_rows, ]
  # mean for drug/feature combination
  my_mean <- mean(my_long_data$local_importance)
  my_mean
}

# reformat the mean table and write it into table
export_mean_in_table <- function(md, my_version, cancer_type) {
  md_export <- md %>%
    rownames_to_column("drug") %>%
    select(drug, everything())

  filename <- paste0(outdirs[[my_version]],
                     cancer_type,
                     "_",
                     my_version,
                     "__drug_feature_mean_local_importance.txt")
  write.table(x = md_export,
              file = filename,
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = T)
}

# with mean of local importance value over all clones/clusters per feature/drug combination (see above)
# take absolute value
# calculate mean of those absolute values per feature (later shown as diamond in violin)
# gives out the rank as well
get_means_summary <- function(md) {
  md %>%
    mutate(across(everything(), abs)) %>% # Convert all values to absolute
    summarise(across(everything(), \(x) mean(x, na.rm = TRUE))) %>% # Compute mean per column
    pivot_longer(
      cols = everything(),
      names_to = "feature",
      values_to = "mean_per_feature"
    ) %>%
    mutate(rank_mean_per_feature = min_rank(mean_per_feature)) # Rank the mean values
}

prepare_table_for_violin <- function(md, df_means_summary) {
  md %>%
    rownames_to_column("drug") %>% # Convert row names into a column
    pivot_longer(
      cols = -drug, # Pivot all columns except 'drug'
      names_to = "feature",
      values_to = "mean_value"
    ) %>%
    mutate(mean_value_abs = abs(mean_value)) %>% # Compute absolute mean value
    left_join(df_means_summary, by = "feature") %>% # Merge with mean ranking table
    # reorder FACTORs (not elements)
    mutate(feature = forcats::fct_reorder(feature, -rank_mean_per_feature)) %>% # Reorder factor by rank
    mutate(sign = if_else(mean_value > 0, "red", "blue")) # Assign colours based on sign
}

export_long_mean_in_table <- function(md_long, cancer_type, my_version) {
  filename <- paste0(outdirs[[my_version]],
                     cancer_type,
                     "_",
                     my_version,
                     "_drug_feature_mean_local_importance.summary_long_format.txt")
  write.table(x = md_long,
              file = filename,
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = T)
}

plot_sorted_violin <- function(md_long, my_version, log_scale = FALSE) {
  p <- ggplot(md_long, aes(x = feature, y = mean_value_abs)) +
    geom_violin(fill = "gray80",
                linewidth = .5,
                alpha = .5,
                adjust = 5,
                scale = "width") +
    ggbeeswarm::geom_quasirandom(alpha = 0.7,
                                 groupOnX = T,
                                 size = .5,
                                 aes(color = sign)) +
    guides(color="none") +
    ggtitle(my_version) +
    xlab("feature") +
    ylab("absolute mean of local importance per drug/feature") +
    scale_x_discrete(guide = guide_axis(angle = 55)) +
    theme_bw() +
    theme(axis.text.x = element_text(size = 8),
          axis.title = element_text(size = 11, face = "bold"),
          plot.title = element_text(size = 20, face = "bold"),
          strip.text = element_text(size = 10, face = "bold"),
          plot.margin = margin(1,1,1.2,1.2, "cm")) +
    geom_point(
      aes(x = feature, y = mean_per_feature),
      color = "blue3",
      size = 1,
      shape = 23
    )
  if (log_scale) {
    p <- p + scale_y_continuous(trans = "log10", labels = scales::comma)
  }

  return(p)
}

save_sorted_violin <- function(plot, outdir, cancer_type, my_version, width, height, format = "pdf", summary_stat, note = ""){
  filename <- paste0(outdir,
                     cancer_type,
                     "_",
                     my_version,
                     "__drug_feature_",
                     summary_stat,
                     "_local_importance.violin",
                     note,
                     ".",
                     format)
  ggsave(filename = filename,
         plot = plot,
         dpi = 300,
         width = width,
         height = height,
         units = "cm")
}

plot_line_ranks <- function(md_long, log_scale=FALSE, vlines=FALSE, values_vlines=c(20, 50, 80)) {
  p <- ggplot(md_long, aes(x = rank_mean_value_abs_per_drug,
                                   y = mean_value_abs,
                                   colour = drug)) +
    geom_point(size = 0.2) +
    geom_line()

  if (log_scale) {
    p <- p + scale_y_continuous(trans = "log10", labels = scales::comma)
  }
  if (vlines){
    vline_df <- data.frame(x = values_vlines, label = as.character(values_vlines))
    p <- p + geom_vline(xintercept = vline_df$x,
                        color = "blue",
                        linetype="dotted",
                        linewidth = 1) +
      geom_text(data = vline_df, aes(x = x, y = max(md_long$mean_value_abs), label = label),
                vjust = -0.5, hjust = -.1, color = "blue")
  }
  return(p)
}

# other -------------------------------------------------------------------


# plot the heatmap
plot_heatmap <- function(md, scale, number_clusters, drug_info, cancer_type, my_version, fontsize_col, summary_stat) {
  # create clustering first. For that, create heatmap object
  p_object <- pheatmap(mat = md,
                       scale = scale,
                       cutree_rows = number_clusters,
                       silent = TRUE)
  # get clusters
  row_clusters <- cutree(p_object$tree_row, k = number_clusters)

  # create dataframe for annotation row
  # rownames are included
  annotation_row <- data.frame(Cluster = factor(row_clusters))

  annotation_row$drug <- rownames(annotation_row)
  annotation_row <- dplyr::full_join(annotation_row,
                                     drug_info,
                                     by = join_by(drug == orig.name))
  rownames(annotation_row) <- annotation_row$drug
  # select columns to show in annotation, make sure they are factors
  annotation_row <- annotation_row %>%
    select(Cluster, drug_target_group) %>%
    mutate(across(everything(), as.character)) %>%
    replace_na(list(drug_target_group = "NA")) %>%
    mutate(across(everything(), as.factor))

  annotation_colors <- lapply(annotation_row, function(column) {
    levels <- unique(column)  # Get unique levels
    colors <- my_colours[1:length(levels)]  # Take the first n colours from the my_colours vector
    setNames(colors[seq_along(levels)], levels)  # Map levels to colours
  })

  heatmap_header <- paste0(cancer_type, ", ", my_version, ", scaling = ", scale, ", ", summary_stat, " per feature/drug pair over all clones/clusters")

  p <- pheatmap(mat = md,
                scale = scale,
                angle_col = 45,
                treeheight_row = 100,
                cutree_rows = number_clusters,
                annotation_row = annotation_row,
                annotation_colors = annotation_colors,
                main = heatmap_header,
                fontsize_col = fontsize_col)
  p
}

# save heatmap plot to PNG or PDF file
save_heatmap <- function(plot, outdir, cancer_type, my_version, scale, width, height, format = "pdf", summary_stat, note=""){
  filename <- paste0(outdir,
                     cancer_type,
                     "_",
                     my_version,
                     "__drug_feature_",
                     summary_stat,
                     "_local_importance.heatmap.scale_",
                     scale,
                     ".",
                     format)
  if (note != "") {
    filename <- paste0(outdir,
                       cancer_type,
                       "_",
                       my_version,
                       "__drug_feature_",
                       summary_stat,
                       "_local_importance.heatmap.scale_",
                       scale,
                       ".",
                       note,
                       ".",
                       format)
  }
  ggsave(filename = filename,
         plot = plot,
         dpi = 300,
         width = width,
         height = height,
         units = "cm")
}

# plot the line plot
save_lineplot <- function(plot, outdir, cancer_type, my_version, format = "pdf", summary_stat, note=""){
  filename <- paste0(outdir,
                     cancer_type,
                     "_",
                     my_version,
                     "__drug_feature_",
                     summary_stat,
                     "_full_line_plot.",
                     format)
  if (note != "") {
    filename <- paste0(outdir,
                       cancer_type,
                       "_",
                       my_version,
                       "__drug_feature_",
                       summary_stat,
                       "_full_line_plot.",
                       note,
                       ".",
                       format)
  }
  ggsave(filename = filename,
         plot = plot,
         dpi = 300,
         width = 50,
         height = 20,
         units = "cm")
}

# hallmark-specific -------------------------------------------------------

shorten_hallmark_pathway_names <- function(x) {
  gsub("^HALLMARK_(.+)$", "\\1", x, perl = TRUE)
}

# violin plot per drug/pathway --------------------------------------------

get_means_for_violin <- function(md_long, drug) {
  md_long %>%
    group_by(feature) %>%
    summarise(mean_per_feature = mean(local_importance, na.rm = TRUE)) %>%
    mutate(mean_per_feature_abs = abs(mean_per_feature)) %>%
    arrange(desc(mean_per_feature_abs)) %>%
    mutate(rank = seq_along(mean_per_feature_abs)) %>%
    mutate(drug = drug)
}


# variance is more important than mean?
get_stats_per_feature <- function(md_long, drug) {
  md_long %>%
    group_by(feature) %>%
    summarise(var_per_feature = var(local_importance,na.rm = TRUE),
              mean_per_feature = mean(local_importance, na.rm = TRUE),
              mean_per_feature_of_absolute_values = mean(abs(local_importance), na.rm = TRUE),
              total_sum_of_squares = sum((local_importance - mean(local_importance))^2, na.rm = TRUE)) %>%
    #mutate(mean_per_feature_abs = abs(mean_per_feature)) %>%
    # sort by variance
    arrange(desc(var_per_feature)) %>%
    mutate(rank = seq_along(var_per_feature)) %>%
    mutate(drug = drug) # add drug name to md_long table (before later combining values)
}

# variance is more important than mean?
get_stats_per_feature_general <- function(md_long, drug) {
  md_long %>%
    group_by(feature) %>%
    summarise(var_per_feature = var(value,na.rm = TRUE),
              mean_per_feature = mean(value, na.rm = TRUE),
              mean_per_feature_of_absolute_values = mean(abs(value), na.rm = TRUE),
              total_sum_of_squares = sum((value - mean(value))^2, na.rm = TRUE)) %>%
    #mutate(mean_per_feature_abs = abs(mean_per_feature)) %>%
    # sort by variance
    arrange(desc(var_per_feature)) %>%
    mutate(rank = seq_along(var_per_feature)) %>%
    mutate(drug = drug) # add drug name to md_long table (before later combining values)
}

plot_violin_of_local_importance_values <- function(md_long = md_long) {
  ggplot(md_long, aes(x = feature, y = local_importance)) +
    geom_violin(fill = "lightskyblue2", linewidth = .5, alpha = .5, adjust = 5, scale = "width") +
    ggbeeswarm::geom_quasirandom(alpha = 0.4, groupOnX = T, size = .5) +
    ggtitle(attrs$dim1_drugs[drug]) +
    scale_x_discrete(guide = guide_axis(angle = 55)) +
    theme(axis.text.x = element_text(size = 8),
          axis.title = element_text(size = 14, face = "bold"),
          plot.title = element_text(size = 20, face = "bold"),
          strip.text = element_text(size = 10, face = "bold"),
          plot.margin = margin(1,1,1.2,1.2, "cm"))
}


plot_violin_of_local_importance_with_stats <- function(md_long = md_long) {
  ggplot(md_long, aes(x = feature, y = local_importance, fill = beta_sign)) +
    geom_violin(linewidth = .5, alpha = .5, adjust = 5, scale = "width") +
    ggbeeswarm::geom_quasirandom(alpha = 0.4, groupOnX = T, size = .5) +
    ggtitle(attrs$dim1_drugs[drug]) +
    scale_x_discrete(guide = guide_axis(angle = 55)) +
    theme(axis.text.x = element_text(size = 8),
          axis.title = element_text(size = 14, face = "bold"),
          plot.title = element_text(size = 20, face = "bold"),
          strip.text = element_text(size = 10, face = "bold"),
          plot.margin = margin(1,1,1.2,1.2, "cm")) +
    geom_point(
      aes(x = feature, y = var_per_feature, colour = "Variance"),
      size = 1.5,
      shape = 18
    ) +
    geom_point(
      aes(x = feature, y = mean_per_feature_of_absolute_values, colour = "Mean of absolute local importance values"),
      size = 1.5,
      shape = 18
    ) +
    scale_color_manual(
      values = c("Variance" = "yellow2", "Mean of absolute local importance values" = "red2")
    ) #+
    #geom_point(
    #  aes(x = feature, y = total_sum_of_squares),
    #  color = "pink",
    #  size = 1,
    #  shape = 23
    #)
}


plot_violin_of_value_with_stats <- function(md_long = md_long) {
  ggplot(md_long, aes(x = feature, y = value)) +
    geom_violin(linewidth = .5, alpha = .5, adjust = 5, scale = "width") +
    ggbeeswarm::geom_quasirandom(alpha = 0.4, groupOnX = T, size = .5) +
    ggtitle(attrs$dim1_drugs[drug]) +
    scale_x_discrete(guide = guide_axis(angle = 55)) +
    theme(axis.text.x = element_text(size = 8),
          axis.title = element_text(size = 14, face = "bold"),
          plot.title = element_text(size = 20, face = "bold"),
          strip.text = element_text(size = 10, face = "bold"),
          plot.margin = margin(1,1,1.2,1.2, "cm")) +
    geom_point(
      aes(x = feature, y = var_per_feature, colour = "Variance"),
      size = 1.5,
      shape = 18
    ) +
    geom_point(
      aes(x = feature, y = mean_per_feature, colour = "Mean"),
      size = 1.5,
      shape = 18
    ) +
    scale_color_manual(
      values = c("Variance" = "yellow2", "Mean" = "red2")
    ) #+
  #geom_point(
  #  aes(x = feature, y = total_sum_of_squares),
  #  color = "pink",
  #  size = 1,
  #  shape = 23
  #)
}



# per drug
save_violin_plot_per_drug <- function(plot, outdir, my_version, cancer_type, drug, width, height, format = "png", note = ""){
  filename <- paste0(outdir,
                     cancer_type,
                     "_",
                     my_version,
                     "__",
                     drug,
                     "_",
                     "drug_feature_violin_plot_local_importance",
                     if (note != "") paste0("_", note),
                     ".",
                     format)
  ggsave(filename = filename,
         plot = plot,
         dpi = 300,
         width = width,
         height = height,
         units = "cm")
}

# per pathway
save_violin_plot_per_pathway <- function(plot, outdir, my_version, cancer_type, pathway, width, height, format = "png", note = ""){
  filename <- paste0(outdir,
                     cancer_type,
                     "_",
                     my_version,
                     "__",
                     pathway,
                     "_",
                     "drug_feature_violin_plot_local_importance",
                     if (note != "") paste0("_", note),
                     ".",
                     format)
  ggsave(filename = filename,
         plot = plot,
         dpi = 300,
         width = width,
         height = height,
         units = "cm")
}

# reformat the mean table and write it into table
export_feature_summary <- function(summary_per_drug, topX, outdir, cancer_type, my_version) {
  full_md_summary <- bind_rows(summary_per_drug) %>%
    group_by(feature) %>%
    summarise(
      all_drugs = stringr::str_c(drug, collapse = ","),
      all_ranks = str_c(rank, collapse = ","),
      #number_of_drugs = n(),
      rank_sums = sum(rank, na.rm = TRUE),
      mean_rank = round(mean(rank, na.rm = TRUE), 1),
      mean_total_sum_of_squares = round(mean(total_sum_of_squares), 3),
      mean_absolute_mean = round(mean(mean_per_feature_of_absolute_values), 3),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_total_sum_of_squares))

  filename <- paste0(outdir,
                     cancer_type,
                     "_",
                     my_version,
                     "_features_local_importance_summarised.txt")
  write.table(x = full_md_summary,
              file = filename,
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = T)
  head(full_md_summary, 10)
}
