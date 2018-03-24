set nocompatible              " be iMproved, required
filetype off                  " required

set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

Plugin 'scrooloose/nerdtree'

call vundle#end()            " required
filetype plugin indent on    " required





" NERDtree"
let NERDTreeWinPos='left'
let NERDTreeWinSize=21
let NERDTreeChDirMode=1
autocmd VimEnter * NERDTree 
let NERDTreeIgnore = ['\.pyc$']


set hlsearch
"设置字符集
set encoding=utf-8
set fileencodings=ucs-bom,utf-8,chinese,prc,taiwan,latin-1,gbk,ucs-bom,cp936
set fileencoding=utf-8
let &termencoding=&encoding
" 解决菜单乱码
source $VIMRUNTIME/delmenu.vim
source $VIMRUNTIME/menu.vim


"背景颜色
"colorscheme torte 
"colorscheme desert 
colorscheme default
"代码高亮
syntax enable
syntax on
"不备份
set nobackup
"显示行号
set nu!
"显示 bracets 配对
set showmatch
"启动时大小
"set lines=110
"set columns=240
"winpos 0 0
"不自动换行
set nolinebreak
set wrap
"历史数
set history=1024
"设置自动缩进
set ai
"将tab转换为空格
set expandtab

" backspace
set bs=2
" scrolloff
set scrolloff=5

" python
"au BufNewFile,BufRead *.py setl tabstop=4 expandtab shiftwidth=4 softtabstop=4 et
" cpp
"autocmd BufRead,BufNewFile *.c,*.C,*.H,*.h set filetype=cpp
"au BufNewFile,BufRead *.C setl tabstop=2 expandtab shiftwidth=2 softtabstop=2 et
"au BufNewFile,BufRead *.c setl tabstop=2 expandtab shiftwidth=2 softtabstop=2 et
"au BufNewFile,BufRead *.cpp setl tabstop=2 expandtab shiftwidth=2 softtabstop=2 et
"autocmd filetype cpp setl tabstop=2 expandtab shiftwidth=2 softtabstop=2 et
" html
"au BufNewFile,BufRead *.html setl tabstop=2 expandtab shiftwidth=2 softtabstop=2 et
" txt
"au BufNewFile,BufRead *.txt setl tabstop=2 expandtab shiftwidth=2 softtabstop=2 et



" GCC
func! CompileGpp()
    exec "w"
    let compilecmd="!g++ "
    let compileflag="-o %< "
    if search("mpi\.h") != 0
        let compilecmd = "!mpic++ "
    endif
    if search("glut\.h") != 0
        let compileflag .= " -lglut -lGLU -lGL "
    endif
    if search("cv\.h") != 0
        let compileflag .= " -lcv -lhighgui -lcvaux "
    endif
    if search("omp\.h") != 0
        let compileflag .= " -fopenmp "
    endif
    if search("math\.h") != 0
        let compileflag .= " -lm "
    endif
    exec compilecmd." % ".compileflag
endfunc

"
map <F2> :NERDTreeToggle<CR>
map <F5> :diffupdate<CR>

"Cool files 
au BufNewFile,BufRead *.cl set filetype=cool
" Cuda
au BufNewFile,BufRead *.cu set ft=cuda
if (&filetype == "objc")
  set tabstop=2 softtabstop=2 shiftwidth=2 noexpandtab
endif

au FileType make setlocal noexpandtab tabstop=4 
au FileType html setlocal noexpandtab tabstop=2 shiftwidth=2 

au BufNewFile,BufRead *.nasm set ft=nasm et ts=4  sw=4

" F8 --> Hex editor
noremap <F8> :call HexMe()<CR>

let $in_hex=0
function HexMe()
    set binary
    set noeol
    if $in_hex>0
        :%!xxd -r
        let $in_hex=0
    else
        :%!xxd
        let $in_hex=1
    endif
endfunction




"" compile
""map <F9> :! python '%'<cr>
"autocmd filetype python nnoremap <F9> :w <bar> exec '!python '.shellescape('%')<CR>
"autocmd filetype c nnoremap <F9> :w <bar> exec '!gcc '.shellescape('%').' -o '.shellescape('%:r').' && ./'.shellescape('%:r')<CR>
"autocmd filetype cpp nnoremap <F9> :w <bar> exec '!g++ '.shellescape('%').' -std=c++11 -o '.shellescape('%:r').' && ./'.shellescape('%:r')<CR>
"autocmd filetype cpp nnoremap <F10> :w <bar> exec '!g++ '.shellescape('%').' -std=c++11 -o '.shellescape('%:r').' &&  cat temp.txt \| ./'.shellescape('%:r')<CR>

" ACM style
func! ACM()
    exec "w"
    exec "! g++ % --std=c++11 && cat temp.txt \| .\/a.exe"
endfunc
map <F8> :call HexMe()<CR>
"map <F10> :! g++ % & ./a.exe <CR>


" execute pathogen#infect()
" call pathogen#helptags()


augroup NERD
        au!
        autocmd VimEnter * NERDTree
        autocmd VimEnter * wincmd p
augroup END


function Compile_and_run()
    exec 'w'
    if &filetype == 'c'
        exec "! gcc % -o %<; time ./%<"
    elseif &filetype == 'cpp'
       exec "! g++ -std=c++11 % -o %<; time ./%<"
    elseif &filetype == 'java'
       exec "! javac %; time java %<"
    elseif &filetype == 'sh'
       exec "! bash %"
    elseif &filetype == 'python'
       exec "! python3 %"
    endif
endfunction

    
" Quick run via <F5>
nnoremap <F9> :call Compile_and_run() <CR>

filetype indent on
