{
  description = "SC4000 IMC 2025 clean-room ML development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          nodejs_22
          git
          tmux
          unzip
          zip
          jq
          curl
          cacert
          pkg-config
        ];
        # Common shared libraries used by Python wheels on NixOS.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.glib
          pkgs.libGL
          pkgs.openssl
        ];
        shellHook = ''
          export PATH="$HOME/.local/bin:$PATH"
          export UV_LINK_MODE=copy
          echo "SC4000 clean-room shell ready (Python $(python --version 2>&1))."
          echo "Next: bash scripts/setup.sh"
        '';
      };
    };
}
