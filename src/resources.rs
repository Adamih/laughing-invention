use std::io::{BufReader, Cursor};

use anyhow::Context;
use cfg_if::cfg_if;
use wgpu::util::DeviceExt;

use crate::{
    model::{self, Mesh},
    texture,
};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let data = reqwest::get(url)
                .await?
                .bytes()
                .await?
                .to_vec();
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let data = std::fs::read(path)?;
        }
    }

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        materials.push(model::Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}

pub async fn load_gltf(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::GLTFModel> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            unimplemented!()
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
        }
    }
    let gltf = gltf::Gltf::open(path)?;

    let accessors = gltf.accessors().collect::<Vec<_>>();

    let mut meshes = Vec::new();
    for gltf_mesh in gltf.meshes() {
        let name = gltf_mesh.name().unwrap_or("unnamed").to_string();
        let mut primitives = Vec::new();
        for gltf_primitive in gltf_mesh.primitives() {
            let gltf_primitive_indices = gltf_primitive
                .indices()
                .context("primitive has no indices")
                .unwrap()
                .index();
            let accessor = accessors
                .get(gltf_primitive_indices)
                .context("accessor not found")
                .unwrap();
            let view = accessor.view().context("accessor has no view").unwrap();
            let buffer = view.buffer();

            let positions = gltf_primitive
                .attributes()
                .find(|(semantic, _)| semantic == &gltf::mesh::Semantic::Positions);
            let normals = gltf_primitive
                .attributes()
                .find(|(semantic, _)| semantic == &gltf::mesh::Semantic::Normals);
            let tex_coords = gltf_primitive
                .attributes()
                .find(|(semantic, _)| semantic == &gltf::mesh::Semantic::TexCoords(0));

            todo!("Build vertex buffer using PrimitiveVertex struct and bytemuck");

            let index_buffer = match buffer.source() {
                gltf::buffer::Source::Bin => unimplemented!(),
                gltf::buffer::Source::Uri(uri) => {
                    let binary_file = load_binary(uri)
                        .await
                        .context("binary file not found")
                        .unwrap();
                    // Get buffer slice from view spec and load it into a buffer
                    let buffer_slice = binary_file
                        .as_slice()
                        .get(view.offset()..(view.offset() + view.length()))
                        .context("buffer slice not found")
                        .unwrap();
                    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("{:?} Buffer", file_name)),
                        contents: buffer_slice,
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    buffer
                }
            };

            primitives.push(model::GLTFPrimitive {
                vertex_buffer: todo!(),
                index_buffer: todo!(),
                num_elements: todo!(),
                material: todo!(),
            })
        }

        meshes.push(model::GLTFMesh { name, primitives })
    }

    todo!();
}
