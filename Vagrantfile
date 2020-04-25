# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|

  config.vm.box = "ubuntu/xenial64"
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "8196"
  end

  config.vm.provision "shell", inline: <<-SHELL
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update

    apt-get install -y python3.6 python3.6-dev
    apt-get install -y cmake g++ python3-setuptools
    apt-get install -y libboost-context-dev libboost-program-options-dev libboost-filesystem-dev doxygen graphviz-dev
    apt-get install -y python3-pip
    
    python3.6 -m pip install --upgrade pip
    python3.6 -m pip install numpy networkx dgl cython torch comet_ml
    cd /vagrant/pysimgrid && bash ./get_simgrid.sh
    cd /vagrant/pysimgrid && python3.6 setup.py install --user
  SHELL

end