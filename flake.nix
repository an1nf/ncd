{
  description = "ncd-threat: Normalized Compression Distance for malware-family and network-anomaly classification";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        manifest = (pkgs.lib.importTOML ./Cargo.toml).workspace.package;

        # Python with the evaluation stack; the Rust bindings are added with
        # `maturin develop` inside the dev shell.
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy pandas scikit-learn matplotlib
        ]);

        ncd = pkgs.rustPlatform.buildRustPackage {
          pname = "ncd";
          inherit (manifest) version;
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              let base = baseNameOf path; in
              !(builtins.elem base [ "target" "demo-out" "results" ".direnv" ]);
          };
          cargoLock.lockFile = ./Cargo.lock;
          # the PyO3 crate is built by maturin, not here
          cargoBuildFlags = [ "-p" "ncd-cli" ];
          cargoTestFlags = [ "-p" "ncd-core" "-p" "ncd-cli" ];
          meta = {
            description = manifest.description or "NCD-based threat classifier";
            license = pkgs.lib.licenses.mit;
            mainProgram = "ncd";
          };
        };
      in
      {
        packages.default = ncd;
        packages.ncd = ncd;

        apps.default = flake-utils.lib.mkApp { drv = ncd; };

        # `nix flake check` builds the CLI and runs the Rust test-suite.
        checks.default = ncd;

        devShells.default = pkgs.mkShell {
          inputsFrom = [ ncd ];
          packages = [
            pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt pkgs.rust-analyzer
            pkgs.maturin pythonEnv
          ];
          shellHook = ''
            echo "ncd-threat: cargo build --release; maturin develop --release -m crates/ncd-py/Cargo.toml"
          '';
        };
      });
}
