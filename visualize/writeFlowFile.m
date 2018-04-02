function writeFlowFile(img, filename)

% writeFlowFile writes a 2-band image IMG into flow file FILENAME 

%   According to the c++ source code of Daniel Scharstein 
%   Contact: schar@middlebury.edu

%   Author: Deqing Sun, Department of Computer Science, Brown University
%   Contact: dqsun@cs.brown.edu
%   $Date: 2007-10-31 15:36:40 (Wed, 31 Oct 2006) $

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

TAG_STRING = 'PIEH';    % use this when WRITING the file

% sanity check
if isempty(filename) == 1
    error('writeFlowFile: empty filename');
end;

idx = findstr(filename, '.');
idx = idx(end);             % in case './xxx/xxx.flo'

if length(filename(idx:end)) == 1
    error('writeFlowFile: extension required in filename %s', filename);
end;

if strcmp(filename(idx:end), '.flo') ~= 1    
    error('writeFlowFile: filename %s should have extension ''.flo''', filename);
end;

[height width nBands] = size(img);

if nBands ~= 2
    error('writeFlowFile: image must have two bands');    
end;    

fid = fopen(filename, 'w');
if (fid < 0)
    error('writeFlowFile: could not open %s', filename);
end;

% write the header
fwrite(fid, TAG_STRING); 
fwrite(fid, width, 'int32');
fwrite(fid, height, 'int32');

% arrange into matrix form
tmp = zeros(height, width*nBands);

tmp(:, (1:width)*nBands-1) = img(:,:,1);
tmp(:, (1:width)*nBands) = squeeze(img(:,:,2));
tmp = tmp';

fwrite(fid, tmp, 'float32');

fclose(fid);
