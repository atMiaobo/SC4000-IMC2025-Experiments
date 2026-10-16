{
  description = "NixOS dev shell for the IMC25 adaptive graph project";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv
          git
          git-lfs
          curl
          wget
          unzip
          pkg-config
          gcc13
          cmake
          ninja
          zlib
          glib
          libglvnd
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.glib
            pkgs.libglvnd
          ]}:''${LD_LIBRARY_PATH:-}

          export PROJECT_ROOT="$PWD"
          export TORCH_HOME="$PWD/models/torch"
          export HF_HOME="$PWD/models/hf"
          export HF_HUB_CACHE="$PWD/models/hf/hub"
          export OMP_NUM_THREADS=1
          export MKL_NUM_THREADS=1

          echo "IMC25 adaptive-graph dev shell"
          echo "Python: $(python --version 2>&1)"
          echo "TORCH_HOME=$TORCH_HOME"
        '';
      };
    };
}
