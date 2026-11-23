# OpenCL headers

The three files in `CL/` are unmodified Khronos OpenCL API headers from
the `v2024.10.24` release. The original Apache 2.0 license is included as
`LICENSE`. Upstream: https://github.com/KhronosGroup/OpenCL-Headers

Bundling headers avoids a separate SDK-header download during a build.
The application targets the OpenCL 1.2 API and OpenCL C 1.2.
An installed OpenCL loader and a GPU or CPU device runtime are still needed.
