clc;
clear;
% close all;

%%read an image
cd '../42'
img = imread('42.jpg');
h_img = size(img, 1);
w_img = size(img, 2);
stride = 16;

x_of_interest = 790; % 19, (333, 106), (567, 312), 69 (396, 517), (312, 355), 8, (629, 505), (849, 541), 67, (242, 472), (790, 383)
y_of_interest = 505; % 70, (519, 286), (987, 328) 23, (873, 363), (429, 400), (181, 610), 5, (748, 330), 32 (742, 562), (655, 301) 61 (745, 376) (549, 587)
% 42 (777, 537) (790 505) 56 (858, 289)
feature_x = floor(x_of_interest / stride);
feature_y = floor(y_of_interest / stride);

figure;
imshow(img)

%%load data
att1 = load('npatt1.mat');
att2 = load('npatt2.mat');
att3 = load('npatt3.mat');
att4 = load('npatt4.mat');
att1 = att1.npatt1;
att2 = att2.npatt2;
att3 = att3.npatt3;
att4 = att4.npatt4;

offset1 = load('npoffset1.mat');
offset2 = load('npoffset2.mat');
offset3 = load('npoffset3.mat');
offset4 = load('npoffset4.mat');
offset1 = offset1.npoffset1;
offset2 = offset2.npoffset2;
offset3 = offset3.npoffset3;
offset4 = offset4.npoffset4;
h = size(offset1, 3);
w = size(offset1, 4);

dial_list = [1, 4, 8, 12];

offset_list = zeros(4, size(offset1, 1), size(offset1, 2), size(offset1, 3), size(offset1, 4));
offset_list(1,:,:,:,:) = offset1;
offset_list(2,:,:,:,:) = offset2;
offset_list(3,:,:,:,:) = offset3;
offset_list(4,:,:,:,:) = offset4;

att_position = feature_x * (size(offset1, 3) - feature_y);


list_att1 = squeeze(att1(:,:,att_position,:));
list_att2 = squeeze(att2(:,:,att_position,:)); 
list_att3 = squeeze(att3(:,:,att_position,:));
list_att4 = squeeze(att4(:,:,att_position,:));
list_att = [list_att1; list_att2; list_att3; list_att4];

list_points = [];
index_ = 1;
%%load the points
for i= 1 : 4
    offset_ = offset_list(i,:,:,:,:);
    
    offset_curr = squeeze(offset_(1,:,:,feature_y, feature_x));
    list_points(index_, :) = [feature_x-dial_list(i)+offset_curr(2), feature_y-dial_list(i)+offset_curr(1)];
    list_points(index_ + 1, :) = [feature_x+offset_curr(4), feature_y-dial_list(i)+offset_curr(3)];             
    list_points(index_ + 2, :) = [feature_x+dial_list(i)+offset_curr(6), feature_y-dial_list(i)+offset_curr(5)];

    list_points(index_ + 3, :) = [feature_x-dial_list(i)+offset_curr(8), feature_y+offset_curr(7)];
    list_points(index_ + 4, :) = [feature_x+offset_curr(10), feature_y+offset_curr(9)];
    list_points(index_ + 5, :) = [feature_x+dial_list(i)+offset_curr(12), feature_y+offset_curr(11)];

    list_points(index_ + 6, :) = [feature_x-dial_list(i)+offset_curr(14), feature_y+dial_list(i)+offset_curr(13)];
    list_points(index_ + 7, :) = [feature_x+offset_curr(16), feature_y+dial_list(i)+offset_curr(15)];
    list_points(index_ + 8, :) = [feature_x+dial_list(i)+offset_curr(18), feature_y+dial_list(i)+offset_curr(17)];

    index_ = index_ + 9;
end

hold on;
A=colormap('jet');
scatter(list_points(:,1) * stride, list_points(:,2) * stride, 70, list_att, 'filled'); %A(15,:), 'MarkerFaceColor', A(15,:)); % , 'MarkerFaceAlpha',0.2, 'MarkerEdgeAlpha',0.2
colormap('parula');
colorbar
hold on;
scatter(x_of_interest, y_of_interest, 90, 'r', 'MarkerFaceColor', 'r');

% points = list_points;
% affinities = rand(9,1);
% hold on;
% 
% for i=1:size(list_points,1)
%     x = list_points(i,1);
%     y = list_points(i,2);
%     opacity = affinities(i);
%     scatter(x * stride, y * stride, 'b', 'MarkerFaceColor', 'c', 'MarkerFaceAlpha', opacity);
%     hold on
% end



% new_att1 = zeros(h-2, w-2);
% for i=2:h-1
%     for j=2:w-1
%         a1 = att1(i-1, j-1, 9);
%         a2 = att1(i, j-1, 8);
%         a3 = att1(i+1, j-1, 7);
%         a4 = att1(i-1, j, 6);
%         a5 = att1(i, j, 5);
%         a6 = att1(i+1, j, 4);
%         a7 = att1(i-1, j+1, 3);
%         a8 = att1(i, j+1, 2);
%         a9 = att1(i+1, j+1, 1);
%         new_att1(i, j) = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
%     end
% end

%%visualize attention
%att_ = new_att1;
% att_ = reshape(att1(:,:), [h,w]);
% %normalization
% att_ = (att_ - min(att_(:))) / (max(att_(:)) - min(att_(:)));
% att_ = imresize(att_, [h_img, w_img]);
% att_ = uint8(att_ * 255);
% att_ = gray2ind(att_, 256);
% att_hm = ind2rgb(att_, jet(256));
% final_img_ = 0.7*im2double(img) + 0.3*att_hm;
% imshow(final_img_);

