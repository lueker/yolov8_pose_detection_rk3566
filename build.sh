set -e
#每次build的时候把build文件夹删除

TARGET_SOC="rk356x"
GCC_COMPILER=aarch64-linux-gnu

export LD_LIBRARY_PATH=/home/dl2/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/lib
export RK356X_TOOLCHAIN=/home/dl2/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu
GCC_COMPILER=${RK356X_TOOLCHAIN}/bin/aarch64-rockchip1031-linux-gnu

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j8
make install
cd -
