# heatmap for lake.cu

set terminal png

set xrange[0:1]
set yrange[0:1]
set cbrange[-0.4:0.4]

set output 'lake_c.png'
plot 'lake_c.dat' using 1:2:3 with image
