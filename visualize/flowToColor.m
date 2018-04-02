function img = flowToColor(flow, varargin)

%  flowToColor(flow, maxFlow) flowToColor color codes flow field, normalize
%  based on specified value, 
% 
%  flowToColor(flow) flowToColor color codes flow field, normalize
%  based on maximum flow present otherwise 

%   According to the c++ source code of Daniel Scharstein 
%   Contact: schar@middlebury.edu

%   Author: Deqing Sun, Department of Computer Science, Brown University
%   Contact: dqsun@cs.brown.edu
%   $Date: 2007-10-31 18:33:30 (Wed, 31 Oct 2006) $

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

UNKNOWN_FLOW_THRESH = 1e9;
UNKNOWN_FLOW = 1e10;            % 

[height widht nBands] = size(flow);

if nBands ~= 2
    error('flowToColor: image must have two bands');    
end;    

u = flow(:,:,1);
v = flow(:,:,2);

maxu = -999;
maxv = -999;

minu = 999;
minv = 999;
maxrad = -1;


% fix unknown flow
idxUnknown = (abs(u)> UNKNOWN_FLOW_THRESH) | (abs(v)> UNKNOWN_FLOW_THRESH) ;

u(idxUnknown) = 0;
v(idxUnknown) = 0;
max(u(:))
maxu = max(maxu, max(u(:)));
minu = min(minu, min(u(:)));

maxv = max(maxv, max(v(:)));
minv = min(minv, min(v(:)));

rad = sqrt(u.^2+v.^2);
maxrad = max(maxrad, max(rad(:)));

fprintf('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n', maxrad, minu, maxu, minv, maxv);

if isempty(varargin) ==0
    maxFlow = varargin{1};
    if maxFlow > 0
        maxrad = maxFlow;
    end;       
end;

u = u/(maxrad+eps);
v = v/(maxrad+eps);

% compute color

img = computeColor(u, v);  
    
% unknown flow
IDX = repmat(idxUnknown, [1 1 3]);
img(IDX) = 0;