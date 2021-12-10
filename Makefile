all: lines_cols pipeline

lines_cols : lines_cols.cpp
	mpic++ lines_cols.cpp -o lines_cols

pipeline: pipeline.cpp
	mpic++ pipeline.cpp -o pipeline