function img = readFlowFile(filename)

% readFlowFile read a flow file FILENAME into 2-band image IMG 

%   According to the c++ source code of Daniel Scharstein 
%   Contact: schar@middlebury.edu

%   Author: Deqing Sun, Department of Computer Science, Brown University
%   Contact: dqsun@cs.brown.edu
%   $Date: 2007-10-31 16:45:40 (Wed, 31 Oct 2006) $

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

TAG_FLOAT = 202021.25;  % check for this when READING the file

% sanity check
if isempty(filename) == 1
    error('readFlowFile: empty filename');
end;

idx = findstr(filename, '.');
idx = idx(end);

if length(filename(idx:end)) == 1
    error('readFlowFile: extension required in filename %s', filename);
end;

if strcmp(filename(idx:end), '.flo') ~= 1    
    error('readFlowFile: filename %s should have extension ''.flo''', filename);
end;

fid = fopen(filename, 'r');
if (fid < 0)
    error('readFlowFile: could not open %s', filename);
end;

tag     = fread(fid, 1, 'float32');
width   = fread(fid, 1, 'int32');
height  = fread(fid, 1, 'int32');

% sanity check

if (tag ~= TAG_FLOAT)
    error('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)', filename);
end;

if (width < 1 || width > 99999)
    error('readFlowFile(%s): illegal width %d', filename, width);
end;

if (height < 1 || height > 99999)
    error('readFlowFile(%s): illegal height %d', filename, height);
end;

nBands = 2;

% arrange into matrix form
tmp = fread(fid, inf, 'float32');
tmp = reshape(tmp, [width*nBands, height]);
tmp = tmp';
img(:,:,1) = tmp(:, (1:width)*nBands-1);
img(:,:,2) = tmp(:, (1:width)*nBands);
      
fclose(fid);

