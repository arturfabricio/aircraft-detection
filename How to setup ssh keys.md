
#Generate key
#save as "id_rsa_HPC"
ssh-keygen

Change config
C:\Users\Jensm\.ssh

Host login2.gbar.dtu.dk
  HostName login2.gbar.dtu.dk
  IdentityFile ~/.ssh/id_rsa_HPC
  User s183685
  Port 22


# in bash shell
goto ssh location
cat ~/.ssh/id_rsa_HPC.pub | ssh s183685@login2.gbar.dtu.dk "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"