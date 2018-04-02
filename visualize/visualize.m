flow_file = '../result/result.mat';
load(flow_file);
img = flowToColor(flow);
figure;
imshow(img)