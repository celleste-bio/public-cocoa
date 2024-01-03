PS1="${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\h\[\033[01;0m\] | \[\033[01;34m\]\w\[\033[01;0m\] > "
alias readdir="ls --format=single-column --almost-all --group-directories-first"
alias rd=readdir
source /home/pyenv/bin/activate