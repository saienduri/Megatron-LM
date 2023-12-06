script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="${script_dir%/*}"
checkpoint_dir="${base_dir}/checkpoints/gpt2_345m"

[ -d "${checkpoint_dir}" ] || mkdir -p "${checkpoint_dir}"

pushd .
cd "${checkpoint_dir}"
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
unzip megatron_lm_345m_v0.0.zip
rm megatron_lm_345m_v0.0.zip
popd
