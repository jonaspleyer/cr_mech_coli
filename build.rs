#[cfg(feature = "vtk")]
use cmake::Config;
#[cfg(feature = "vtk")]
use vtk_rs_link::{log, Result, WARN};

// Handle building of cmake project
#[cfg(feature = "vtk")]
fn build_cmake() {
    println!("cargo:rerun-if-changed=vtkRender");
    let mut config = Config::new("vtkRender");

    if std::env::var("CARGO_FEATURE_V094").is_ok_and(|x| x == "1") {
        config.define("VTK094", "1");
    }

    let dst = config.build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=vtkRender");
}

#[cfg(feature = "vtk")]
fn main() -> Result<()> {
    // Exit early without doing anything if we are building for docsrs
    if std::env::var("DOCS_RS").is_ok() {
        return Ok(());
    }

    if let Ok(val) = std::env::var("VERBOSE") {
        if val == "1" || val.to_lowercase() == "true" {
            WARN.store(true, std::sync::atomic::Ordering::Relaxed);
            log!("-- Verbose Logging Enabled");
        }
    }

    // Build cpp project
    build_cmake();

    // Link to VTK
    let modules = include_str!("vtkRender/modules.txt").lines();

    vtk_rs_link::link_cmake_project(modules)?;

    Ok(())
}

#[cfg(not(feature = "vtk"))]
fn main() {}
