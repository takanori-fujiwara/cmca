library(reticulate)
library(rstudioapi)

library(ggplot2)
library(gridExtra)

# helper to convert native numpy array to r's matrix
nparr_to_rmat <- function(nparr) {
  n_rows <- py_to_r(nparr$shape[0])
  n_cols <- py_to_r(nparr$shape[1])
  rmat <- t(matrix(unlist(py_to_r(nparr$tolist())), dim <- c(n_cols, n_rows)))
  return(rmat)
}

## Load python
# If you want to use non-virtual environment Python:
use_python("/usr/local/bin/python3")

# If you want to use virtual environment Python:
# use_virtualenv("/path/to/your/virtualenv")

np <- import("numpy", convert=FALSE)
CMCA <- import("cmca", convert=FALSE)$CMCA

# Congressional Voting Records Data Set
# https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
df <- read.csv("data/house-votes-84.data", header=FALSE)
colnames(df) <- unlist(read.csv("data/house-votes-84.col_names", header=FALSE))

X <- df[, 2:ncol(df)]
y <- df$class

fg <- X[y == "democrat", ]
bg <- X[y == "republican", ]

# alpha = 0 (normal MCA on fg)
# alpha = 10 (contrastive MCA fg vs bg)
# alpha = 'auto' (contrastive MCA with auto selection of alpha)
cmca <- CMCA(n_components=as.integer(2)) # conversion to integer is important

for (alpha in c(0, 10, "auto")) {
  ### cMCA
  auto_alpha <- FALSE
  if (alpha == "auto") {
    alpha <- NULL
    auto_alpha <- TRUE
  } else{
    alpha <- as.numeric(alpha)
  }
  #cmca$fit(np$array(fg), np$array(bg), alpha=alpha, auto_alpha_selection=auto_alpha)
  cmca$fit(fg, bg, alpha=alpha, auto_alpha_selection=auto_alpha)
  
  # row coordinates (cloud of individuals)
  Y_fg_row <- nparr_to_rmat(np$array(cmca$transform(fg, axis='row')))
  Y_bg_row <- nparr_to_rmat(np$array(cmca$transform(bg, axis='row')))
  
  # col coordinates (cloud of categories)
  Y_fg_col <- nparr_to_rmat(np$array(cmca$transform(fg, axis='col')))
  Y_bg_col <- nparr_to_rmat(np$array(cmca$transform(bg, axis='col')))
  
  # cPC loadings
  loadings <- nparr_to_rmat(cmca$loadings)
  
  # category names
  categories <- py_to_r(cmca$categories)
  
  ### Plot the results
  alpha <- py_to_r((cmca$alpha * 100 / 100))
  
  # plot row coordinates
  row_coord_data <- as.data.frame(rbind(Y_fg_row, Y_bg_row))
  colnames(row_coord_data) <- c("x", "y")
  row_coord_data$label <- c(rep("demo", nrow(Y_fg_row)), rep("rep", nrow(Y_bg_row)))
  
  row_coord_plot <- ggplot(data=row_coord_data, aes(x=x, y=y, color=label)) +
    geom_point(alpha=0.8, stroke=0) +
    ggtitle(paste0("cMCA row coords. alpha=", alpha))
  
  # plot col coordinates
  col_coord_data <- as.data.frame(rbind(Y_fg_col, Y_bg_col))
  colnames(col_coord_data) <- c("x", "y")
  col_coord_data$label <- c(rep("demo", nrow(Y_fg_col)), rep("rep", nrow(Y_bg_col)))
  
  col_coord_plot <- ggplot(data=col_coord_data, aes(x=x, y=y, color=label)) +
    geom_point(alpha=0.8, stroke=0)
  
  grid.arrange(row_coord_plot, col_coord_plot, ncol=2)
}
