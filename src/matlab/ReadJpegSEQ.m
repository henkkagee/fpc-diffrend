function [ImageCellArray, headerInfo] = ReadJpegSEQ(fileName,frames)
% -------------------------------------------------------------------------
% Read compressed or uncompressed NorPix image sequence in MATLAB.
% This script can read all frames or a set reading window.
% Index file will be used if available and named as the source file
% (eg. test.seq.idx). Otherwise the script will skip through a compressed
% sequence from the beginning (may take some time).
% 
% INPUTS
%    fileName:       Char containing the full path to the sequence
%    frames:         1x2 double array of beginning and end frame
% OUTPUTS
%    ImageCellArray: Cell array with images and timestamps of all allocated
%                    frames.
%    headerInfo:     Struct with header information (ImageWidth,
%                    ImageHeight, ImageBitDepth, ImageBitDepthReal,
%                    ImageSizeBytes, ImageFormat, AllocatedFrames,
%                    Compression, HeaderVersion, HeaderSize, Description,
%                    TrueImageSize, FrameRate).
% EXAMPLES
%    Read all frames:
%    I = ReadJpegSEQ('C:\test.seq')
% 
%    Read frames 2 to 13:
%    I = ReadJpegSEQ('C:\test.seq',[2 13])
% 
%    Include sequence header information:
%    [I, headerInfo] = ReadJpegSEQ('C:\test.seq',[1 1])
% 
% Last modified 05.11.2021 by Paul Siefert, PhD
% Goethe-University Frankfurt
% siefert@bio.uni-frankfurt.de
% 
% Based on the work of Brett Shoelson (Norpix2MATLAB_MarksMod.m)
% Thanks to NorPix support for providing sequence information.
% 
% This code was tested with Norpix SEQ
% 8-bit monochrome 75% lossy jpeg compression (24.07.2018)
% 8-bit monochrome uncompressed (03.06.2019)
% 8-bit(24) BGR 75% lossy jpeg compressed (04.10.2021)
% Code for RGB included but not tested!
% 
% Please report any bugs and improvement suggestions!
% -------------------------------------------------------------------------

%% Open Sequence & Read Information
fid = fopen(fileName,'r','b'); % open the sequence
assert(fid > 0,['File not found: ' fileName])
endianType = 'ieee-le'; % use little endian machine format ordering for reading bytes

fseek(fid,548,'bof');  % jump to position 548 from beginning
imageInfo = fread(fid,24,'uint32',0,endianType); % read 24 bytes with uint32 precision
headerInfo.ImageWidth = imageInfo(1);
headerInfo.ImageHeight = imageInfo(2);
headerInfo.ImageBitDepth = imageInfo(3);
headerInfo.ImageBitDepthReal = imageInfo(4);
headerInfo.ImageSizeBytes = imageInfo(5);
vals = [0,100,101,200:100:600,610,620,700,800,900];
fmts = {'Unknown','Monochrome','Raw Bayer','BGR','Planar','RGB',...
    'BGRx', 'YUV422', 'YUV422_20', 'YUV422_PPACKED', 'UVY422', 'UVY411', 'UVY444'};
headerInfo.ImageFormat = fmts{vals == imageInfo(6)};
fseek(fid,572,'bof');
headerInfo.AllocatedFrames = fread(fid,1,'ushort',endianType);
fseek(fid,620,'bof');
headerInfo.Compression = fread(fid,1,'uint8',endianType);
% Additional sequence information
fseek(fid,28, 'bof');
headerInfo.HeaderVersion = fread(fid,1,'long',endianType);
fseek(fid,32,'bof');
headerInfo.HeaderSize = fread(fid,4/4,'long',endianType);
fseek(fid,592, 'bof');
DescriptionFormat = fread(fid,1,'long',endianType)';
fseek(fid,36,'bof');
headerInfo.Description = fread(fid,512,'ushort',endianType)';
if DescriptionFormat == 0 %#ok Unicode
    headerInfo.Description = native2unicode(headerInfo.Description);
elseif DescriptionFormat == 1 %#ok ASCII
    headerInfo.Description = char(headerInfo.Description);
end
fseek(fid,580,'bof');
headerInfo.TrueImageSize = fread(fid,1,'ulong',endianType);
fseek(fid,584,'bof');
headerInfo.FrameRate = fread(fid,1,'double',endianType);

% warn if image format is not tested
if strcmp(headerInfo.ImageFormat,'Monochrome')
    fprintf('Proceeding with image format %s.\n', headerInfo.ImageFormat)
elseif strcmp(headerInfo.ImageFormat,'BGR')
    fprintf('Proceeding with image format %s.\n', headerInfo.ImageFormat)
else
    fprintf('Current image format (%s) was not tested and may not work! \n', headerInfo.ImageFormat)
    fprintf('Tested: %s\n', ...
        '8-bit monochrome uncompressed', ...
        '8-bit monochrome 75% lossy jpeg compression', ...
        '8-bit(24) BGR 75% lossy jpeg compression.')
end

%% Reading preparation and error detection

if exist('frames','var') == 1
    % analyze read window input
    if size(frames,2)~=2
        error('False input arguments for frames. Please use 1x2 double array.')
    elseif frames(1) > headerInfo.AllocatedFrames || frames(2) > headerInfo.AllocatedFrames
        error(['Some values of selcted frames (' num2str(frames) ') are above allocated frames of ' num2str(headerInfo.AllocatedFrames) '.'])
    elseif frames(1) > frames(2)
        error(['Value of end frame (' num2str(frames(1)) ') is below first frame (' num2str(frames(2)) ').'])
    end
else
    frames(1:2) = [0 0];
end

% set read window & determine number of frames to read
if frames(1) <= 0, first = 1; else, first = frames(1); end
if frames(2) <= 0, last = headerInfo.AllocatedFrames; else, last = frames(2); end
readAmount = last+1 - first;

if readAmount ~= headerInfo.AllocatedFrames
    loopEnd = readAmount;
else
    loopEnd = headerInfo.AllocatedFrames; % user selected no read window
end
ImageCellArray = cell(readAmount,2); % create an empty cell to store images

% set bit depth for reading
switch headerInfo.ImageBitDepthReal
    case 8
        bitDepth = 'uint8';
    case {10,12,14,16}
        bitDepth = 'uint16'; % not tested
end

% try to open idx file
fidIdx = fopen([fileName '.idx']);
if fidIdx > 0
    fprintf('Idx file found. Using idx information to read frames %d to %d.\n',first,last);
    idxPresent = 1;
else
    fprintf('No idx file found. Reading frames %d to %d.\n',first,last');
    idxPresent = 0;
end

%% Start sequence reading

if headerInfo.Compression == 1 % sequence is compressed
    % disable Warning: JPEG library error (8 bit), "Premature end of JPEG file".
    warning('off','MATLAB:imagesci:jpg:libraryMessage')
    
    if ~idxPresent
        readStart = 1024; % norpix compressed sequences have header size of 1024
        if first > 1
            fprintf('Skipping through frames... ');
            tic
            for i = 1:first-1
                readStart = SkipThroughCompressedSequence(fid, readStart);
            end
            fprintf('completed in %0.1d seconds.\n',toc)
        end
        for i = 1:loopEnd % begin loop
            [Img, readStart] = ReadCompressedFrameWithoutIdx(fid, readStart, bitDepth);
            ImageCellArray(i,1:2) = Img;
        end
        
    elseif idxPresent
        for i = 1:loopEnd
            frame = first-1+i;
            Img = ReadCompressedFrameWithIdx(fid, fidIdx, frame);
            ImageCellArray(i,1:2) = Img;
        end
        fclose(fidIdx);
    end
    
elseif headerInfo.Compression == 0 % sequence is uncompressed
    
    for i = 1:loopEnd
        readStart = ((first-1+i-1)*headerInfo.TrueImageSize)+8192; % set readStart to frame
        ImageCellArray(i,1:2) = ReadUncompressedFrameWithoutIdx(fid, readStart, headerInfo, bitDepth);
    end
end
fclose(fid);
return

%% Reading Functions

function readStart = SkipThroughCompressedSequence(fid, readStart)
% read image buffer size and iterate
fseek(fid,readStart,'bof');
imageBufferSize = fread(fid,1,'uint32','ieee-le');
readStart = readStart+imageBufferSize+8;

function [I, readStart] = ReadCompressedFrameWithoutIdx(fid, readStart, bitDepth)
endianType = 'ieee-le';
% read image buffer size
fseek(fid,readStart,'bof');
imageBufferSize = fread(fid,1,'uint32',endianType);

% read compressed image
readStart = readStart+4;
fseek(fid,readStart,'bof');
SEQ = fread(fid,imageBufferSize,bitDepth,endianType);

% read timestamp
readStart = readStart+imageBufferSize-4;
fseek(fid,readStart,'bof');
time = readTimestamp(fid, endianType);
I{1,2} = time;

% write and read temp image
tempName = '_tmp3.jpg';
tempFile = fopen(tempName,'w');
fwrite(tempFile,SEQ);
fclose(tempFile);
I{1,1} = imread(tempName);

% set read start to after timestamp
readStart = readStart+8;

function I = ReadCompressedFrameWithIdx(fid, fidIdx, frame)
endianType = 'ieee-le';
% read frame using idx buffer size information
if frame == 1
    readStart = 1028;
    fseek(fidIdx,8,'bof');
    imageBufferSize = fread(fidIdx,1,'ulong',endianType);
else
    readStartIdx = frame*24;
    fseek(fidIdx,readStartIdx,'bof');
    readStart = fread(fidIdx,1,'uint64',endianType)+4;
    imageBufferSize = fread(fidIdx,1,'ulong',endianType);
end

fseek(fid,readStart,'bof');
JpegSEQ = fread(fid,imageBufferSize,'uint8',endianType);

% Use > two temp files to prevent fopen errors
if mod(frame,2)
    tempName = '_tmp1.jpg';
else
    tempName = '_tmp2.jpg';
end
tempFile = fopen(tempName,'w');
if tempFile < 0
    tempName = '_tmp3.jpg';
    tempFile = fopen(tempName,'w');
end
fwrite(tempFile,JpegSEQ);
fclose(tempFile);
I{1,1} = imread(tempName);

% read timestamp
readStart = readStart+imageBufferSize-4;
fseek(fid,readStart,'bof');
time = readTimestamp(fid, endianType);
I{1,2} = time;

function I = ReadUncompressedFrameWithoutIdx(fid, readStart, headerInfo, bitDepth)
endianType = 'ieee-le';

% read uncompressed image
fseek(fid,readStart,'bof');
vec = fread(fid, headerInfo.ImageSizeBytes, bitDepth, endianType);

if strcmp(headerInfo.ImageFormat,'Monochrome')
    I{1,1} = uint8(reshape(vec,headerInfo.ImageWidth,headerInfo.ImageHeight)');
elseif strcmp(headerInfo.ImageFormat,'BGR')
    B = reshape(vec(1:3:end,1),headerInfo.ImageWidth,headerInfo.ImageHeight)';
    G = reshape(vec(2:3:end,1),headerInfo.ImageWidth,headerInfo.ImageHeight)';
    R = reshape(vec(3:3:end,1),headerInfo.ImageWidth,headerInfo.ImageHeight)';
    I{1,1} = uint8(cat(3,R,G,B));
elseif strcmp(headerInfo.ImageFormat,'RGB')
    R = reshape(vec(1:3:end,1),headerInfo.ImageWidth,headerInfo.ImageHeight)';
    G = reshape(vec(2:3:end,1),headerInfo.ImageWidth,headerInfo.ImageHeight)';
    B = reshape(vec(3:3:end,1),headerInfo.ImageWidth,headerInfo.ImageHeight)';
    I{1,1} = uint8(cat(3,R,G,B));
end

% read timestamp
readStart = readStart + headerInfo.ImageSizeBytes;
fseek(fid,readStart,'bof');
I{1,2} = readTimestamp(fid, endianType);

function time = readTimestamp(fid, endianType)
imageTimestamp = fread(fid,1,'int32',endianType);
subSec = fread(fid,2,'uint16',endianType);
subSec_str = cell(2,1);
for sS = 1:2
    subSec_str{sS} = num2str(subSec(sS));
    while length(subSec_str{sS})<3
        subSec_str{sS} = ['0' subSec_str{sS}];
    end
end
timestampDateNum = imageTimestamp/86400 + datenum(1970,1,1);
time = [datestr(timestampDateNum) ':' subSec_str{1},subSec_str{2}];
return
