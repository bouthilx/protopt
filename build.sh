# Install pytorch 

PROJECT=$1

PROJECT_PATH=$2

PROTOPT_DIR="$(dirname "$(readlink -f "$0")")"
WHEEL_DIR=$HOME/wheels/$PROJECT

mkdir -p $WHEEL_DIR

module add 'cuda/8.0.44' 'cudnn/7.0' 'gcc/5.4.0' 'imkl/2017.1.132' 'opencv/2.4.13.3' 'python/2.7.13' 'python27-scipy-stack/2017a' 'python27-mpi4py/2.0.0' 'hdf5/1.8.18'

# subprocess32 for smartdispatch
pip wheel --wheel-dir=$WHEEL_DIR subprocess32

# PyMongo for sacred
pip wheel --wheel-dir=$WHEEL_DIR pymongo gitpython

# Pre-create packages to have dependencies ready for protopt-launch
# (Speeds up wheel creations at launch time)
pip wheel --wheel-dir=$WHEEL_DIR ${PROTOPT_DIR}/smartdispatch
pip wheel --wheel-dir=$WHEEL_DIR ${PROTOPT_DIR}/sacred

pip wheel --wheel-dir=$WHEEL_DIR numpy scikit-optimize

# Create protopt whitin a virtualenv to satisfy dependencies
virtualenv $HOME/tmp_protopt_env
source $HOME/tmp_protopt_env/bin/activate

  # Don't pass PROJECT_PATH because we don't want to build it here. It will be
  # completly builth later
  bash ${PROTOPT_DIR}/install_requirements.sh $PROJECT

  pip wheel --wheel-dir=$WHEEL_DIR protopt
deactivate
rm -rf $HOME/tmp_protopt_env

# Project is built here
bash ${PROJECT_PATH}/build.sh
