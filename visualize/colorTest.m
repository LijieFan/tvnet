function colorTest()

%   colorTest() creates a test image showing the color encoding scheme

%   According to the c++ source code of Daniel Scharstein 
%   Contact: schar@middlebury.edu

%   Author: Deqing Sun, Department of Computer Science, Brown University
%   Contact: dqsun@cs.brown.edu
%   $Date: 2007-10-31 20:22:10 (Wed, 31 Oct 2006) $

% Copyright 2007, Deqing Sun.
%
%                         All Rights Reserved
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a
% commercial product is hereby granted without fee, provided that the
% above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of the author and Brown University not be used in
% advertising or publicity pertaining to distribution of the software
% without specific, written prior permission.
%
% THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
% ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 

%% test color pattern of Daniel's c++ code

truerange = 1;
height = 151;
width  = 151;
range = truerange * 1.04;

s2 = round(height/2);

[x y] = meshgrid(1:width, 1:height);

u = x*range/s2 - range;
v = y*range/s2 - range;

img = computeColor(u/truerange, v/truerange);

img(s2,:,:) = 0;
img(:,s2,:) = 0;

figure;
imshow(img);
title('test color pattern');
pause; close;

% test read and write flow
F(:,:,1) = u;
F(:,:,2) = v;
writeFlowFile(F, 'colorTest.flo');
F2 = readFlowFile('colorTest.flo');

u2 = F2(:,:,1);
v2 = F2(:,:,2);

img2 = computeColor(u2/truerange, v2/truerange);

img2(s2,:,:) = 0;
img2(:,s2,:) = 0;

figure; imshow(img2);
title('saved and reloaded test color pattern');
pause; close;

% color encoding scheme for optical flow
img = computeColor(u/range/sqrt(2), v/range/sqrt(2));

img(s2,:,:) = 0;
img(:,s2,:) = 0;

figure;
imshow(img);
title('optical flow color encoding scheme');
pause; close;
