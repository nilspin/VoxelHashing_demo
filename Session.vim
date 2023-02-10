let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd D:\Other_misc\Projects\VoxelHashing_demo
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 Solver.cpp
badd +2 Solver.cu
badd +179 CameraTracking.cpp
badd +1 CameraTrackingUtils.cu
badd +1 Application.cpp
badd +48 Application.h
badd +17 common.h
badd +1 Solver.h
badd +2 CameraTracking.h
badd +1 VoxelUtils.cu
badd +1 SDF_Hashtable.cpp
badd +43 SDF_Hashtable.h
badd +29 VoxelDataStructures.h
badd +1 output.txt
badd +1 ImageProcessingUtils.h
badd +1 ImageProcessingUtils.cpp
badd +0 ImageProcessingUtils.cu
badd +0 CMakeLists.txt
argglobal
%argdel
edit CameraTrackingUtils.cu
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd w
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 140 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 141 + 141) / 282)
exe '3resize ' . ((&lines * 28 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 141 + 141) / 282)
exe '4resize ' . ((&lines * 3 + 34) / 68)
exe 'vert 4resize ' . ((&columns * 141 + 141) / 282)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
230
normal! zo
495
normal! zo
505
normal! zo
let s:l = 4 - ((3 * winheight(0) + 32) / 65)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 0
wincmd w
argglobal
if bufexists("CameraTracking.h") | buffer CameraTracking.h | else | edit CameraTracking.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
39
normal! zo
let s:l = 1 - ((0 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
wincmd w
argglobal
if bufexists("CameraTracking.cpp") | buffer CameraTracking.cpp | else | edit CameraTracking.cpp | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
24
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
153
normal! zo
155
normal! zo
160
normal! zo
201
normal! zo
218
normal! zo
228
normal! zo
let s:l = 243 - ((13 * winheight(0) + 14) / 28)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
243
normal! 07|
wincmd w
argglobal
enew
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
wincmd w
exe 'vert 1resize ' . ((&columns * 140 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 141 + 141) / 282)
exe '3resize ' . ((&lines * 28 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 141 + 141) / 282)
exe '4resize ' . ((&lines * 3 + 34) / 68)
exe 'vert 4resize ' . ((&columns * 141 + 141) / 282)
tabedit Solver.cpp
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 141 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 140 + 141) / 282)
exe '3resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 140 + 141) / 282)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
11
normal! zo
57
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
83
normal! zo
158
normal! zo
let s:l = 1 - ((0 * winheight(0) + 32) / 65)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
wincmd w
argglobal
if bufexists("Solver.h") | buffer Solver.h | else | edit Solver.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
20
normal! zo
35
normal! zo
let s:l = 53 - ((1 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
53
normal! 016|
wincmd w
argglobal
if bufexists("Solver.cu") | buffer Solver.cu | else | edit Solver.cu | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2 - ((1 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 0
wincmd w
exe 'vert 1resize ' . ((&columns * 141 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 140 + 141) / 282)
exe '3resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 140 + 141) / 282)
tabedit Application.cpp
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 141 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 140 + 141) / 282)
exe '3resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 140 + 141) / 282)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
90
normal! zo
137
normal! zo
let s:l = 232 - ((32 * winheight(0) + 32) / 65)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
232
normal! 08|
wincmd w
argglobal
if bufexists("Application.h") | buffer Application.h | else | edit Application.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
27
normal! zo
let s:l = 48 - ((1 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
48
normal! 026|
wincmd w
argglobal
if bufexists("common.h") | buffer common.h | else | edit common.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 17 - ((1 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
17
normal! 022|
wincmd w
exe 'vert 1resize ' . ((&columns * 141 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 140 + 141) / 282)
exe '3resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 140 + 141) / 282)
tabedit SDF_Hashtable.cpp
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 141 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 140 + 141) / 282)
exe '3resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 140 + 141) / 282)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 30 - ((29 * winheight(0) + 32) / 65)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
30
normal! 03|
wincmd w
argglobal
if bufexists("SDF_Hashtable.h") | buffer SDF_Hashtable.h | else | edit SDF_Hashtable.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 44 - ((31 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
44
normal! 0
wincmd w
argglobal
if bufexists("VoxelUtils.cu") | buffer VoxelUtils.cu | else | edit VoxelUtils.cu | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1 - ((0 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
wincmd w
exe 'vert 1resize ' . ((&columns * 141 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 140 + 141) / 282)
exe '3resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 140 + 141) / 282)
tabedit VoxelDataStructures.h
set splitbelow splitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 29 - ((28 * winheight(0) + 32) / 65)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
29
normal! 0
tabedit ImageProcessingUtils.h
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd w
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 140 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 141 + 141) / 282)
exe '3resize ' . ((&lines * 28 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 141 + 141) / 282)
exe '4resize ' . ((&lines * 3 + 34) / 68)
exe 'vert 4resize ' . ((&columns * 141 + 141) / 282)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
12
normal! zo
let s:l = 11 - ((10 * winheight(0) + 32) / 65)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 04|
wincmd w
argglobal
if bufexists("CMakeLists.txt") | buffer CMakeLists.txt | else | edit CMakeLists.txt | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 4 - ((3 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 0
wincmd w
argglobal
if bufexists("ImageProcessingUtils.cu") | buffer ImageProcessingUtils.cu | else | edit ImageProcessingUtils.cu | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
24
normal! zo
32
normal! zo
32
normal! zo
32
normal! zo
32
normal! zo
32
normal! zo
32
normal! zo
32
normal! zo
32
normal! zo
34
normal! zo
51
normal! zo
53
normal! zo
61
normal! zo
61
normal! zo
61
normal! zo
74
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
87
normal! zo
92
normal! zo
107
normal! zo
117
normal! zo
122
normal! zo
132
normal! zo
132
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
132
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
134
normal! zo
let s:l = 86 - ((13 * winheight(0) + 14) / 28)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
86
normal! 04|
wincmd w
argglobal
enew
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=10
setlocal fml=1
setlocal fdn=20
setlocal fen
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 140 + 141) / 282)
exe '2resize ' . ((&lines * 32 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 141 + 141) / 282)
exe '3resize ' . ((&lines * 28 + 34) / 68)
exe 'vert 3resize ' . ((&columns * 141 + 141) / 282)
exe '4resize ' . ((&lines * 3 + 34) / 68)
exe 'vert 4resize ' . ((&columns * 141 + 141) / 282)
tabnext 6
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFc
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
