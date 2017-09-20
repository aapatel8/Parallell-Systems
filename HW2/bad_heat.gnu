# heatmap for lake.cu

set terminal png
#set cbrange[-1:1]
set xrange[0:1]
set yrange[0:1]

#set output 'lake_i_mpi.png'
#plot 'lake_i_mpi.dat' using 1:2:3 with image

set output 'lake_f_0.png'
plot 'lake_f_0.dat' using 1:2:3 with image

set output 'lake_f_1.png'
plot 'lake_f_1.dat' using 1:2:3 with image

set output 'lake_f_2.png'
plot 'lake_f_2.dat' using 1:2:3 with image

set output 'lake_f_3.png'
plot 'lake_f_3.dat' using 1:2:3 with image

