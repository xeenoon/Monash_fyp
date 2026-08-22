# Dependencies

Build/runtime dependencies for the Vulkan renderer. Commands target **Arch
Linux**; adapt package names for other distros.

## Toolchain

| Need            | Arch package                    | Notes                          |
|-----------------|---------------------------------|--------------------------------|
| C11 compiler    | `base-devel` (gcc)              | clang also works               |
| CMake ≥ 3.20    | `cmake`                         |                                |
| `glslc`         | `shaderc`                       | compiles GLSL → SPIR-V         |

## Libraries

| Library     | Arch package                              | Source          | CMake target        |
|-------------|-------------------------------------------|-----------------|---------------------|
| Vulkan      | `vulkan-headers`, `vulkan-icd-loader`     | official repos  | `Vulkan::Vulkan`    |
| GPU driver  | `vulkan-radeon` / `vulkan-intel` / `nvidia-utils` | official repos | (runtime ICD)  |
| SDL3        | `sdl3`                                     | official repos  | `PkgConfig::SDL3`   |
| cglm        | `cglm`                                     | **AUR**         | `cglm::cglm`        |

`cglm` is header-only for our usage (the `glms_*` struct API is `static inline`),
so nothing is linked from it at runtime — CMake only needs its headers.

## Install

Official-repo packages:

```sh
sudo pacman -S --needed base-devel cmake shaderc \
    vulkan-headers vulkan-icd-loader sdl3
# plus your GPU's Vulkan driver, e.g.:
sudo pacman -S vulkan-radeon        # AMD
# sudo pacman -S vulkan-intel       # Intel
# sudo pacman -S nvidia-utils       # NVIDIA
```

cglm from the AUR (you have `yay`):

```sh
yay -S cglm
```

<details>
<summary>Manual AUR install (no helper)</summary>

```sh
git clone https://aur.archlinux.org/cglm.git
cd cglm
makepkg -si          # do NOT run as root; makepkg refuses
```
</details>

This installs cglm's headers and its CMake config, so `find_package(cglm REQUIRED)`
in `CMakeLists.txt` resolves without any extra flags.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/vulkan_cube
```

## Notes

- The renderer targets **Vulkan clip space**: `CMakeLists.txt` defines
  `CGLM_FORCE_DEPTH_ZERO_TO_ONE` (right-handed, 0..1 depth) and the projection
  matrix flips Y (`camera_projection` in `src/camera.c`), since cglm's NDC is
  Y-up while Vulkan's framebuffer Y points down.
