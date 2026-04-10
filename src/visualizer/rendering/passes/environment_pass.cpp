/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "environment_pass.hpp"
#include "core/logger.hpp"
#include <glad/glad.h>

namespace lfs::vis {

    bool EnvironmentPass::shouldExecute(const DirtyMask frame_dirty, const FrameContext& ctx) const {
        (void)frame_dirty;
        if (splitViewEnabled(ctx.settings.split_view_mode)) {
            return false;
        }

        if (!environmentBackgroundEnabled(ctx.settings)) {
            return (frame_dirty & sensitivity()) != 0;
        }

        if (environmentBackgroundUsesTransparentViewerCompositing(ctx.settings)) {
            return true;
        }

        return (frame_dirty & sensitivity()) != 0;
    }

    void EnvironmentPass::execute(lfs::rendering::RenderingEngine&,
                                  const FrameContext& ctx,
                                  FrameResources&) {
        // Always clear the default framebuffer deterministically before present/overlay passes.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);

        glViewport(ctx.viewport_pos.x, ctx.viewport_pos.y, ctx.render_size.x, ctx.render_size.y);
        glClearColor(ctx.settings.background_color.r, ctx.settings.background_color.g,
                     ctx.settings.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!environmentBackgroundEnabled(ctx.settings)) {
            return;
        }

        auto frame_view = ctx.makeFrameView();
        const auto environment_path = resolveEnvironmentPathCached(ctx.settings.environment_map_path);
        if (auto render_result = renderer_.render(
                frame_view,
                environment_path,
                ctx.settings.environment_exposure,
                ctx.settings.environment_rotation_degrees,
                ctx.settings.equirectangular);
            !render_result) {
            // Avoid per-frame log spam when a preset is missing.
            if (render_result.error() != last_environment_error_) {
                last_environment_error_ = render_result.error();
                LOG_DEBUG("Environment background fallback: {}", last_environment_error_);
            }
        } else {
            last_environment_error_.clear();
        }
    }

    std::filesystem::path EnvironmentPass::resolveEnvironmentPathCached(const std::string& path_value) {
        if (path_value == cached_environment_path_value_) {
            return cached_environment_resolved_path_;
        }

        cached_environment_path_value_ = path_value;
        const std::filesystem::path requested(path_value);
        if (requested.empty() || requested.is_absolute()) {
            cached_environment_resolved_path_ = requested;
            return cached_environment_resolved_path_;
        }

        const auto build_assets = std::filesystem::path(VISUALIZER_ASSET_PATH);
        const auto source_assets = std::filesystem::path(VISUALIZER_SOURCE_ASSET_PATH);

        const auto build_candidate = build_assets / requested;
        if (std::filesystem::exists(build_candidate)) {
            cached_environment_resolved_path_ = build_candidate;
        } else {
            cached_environment_resolved_path_ = source_assets / requested;
        }
        return cached_environment_resolved_path_;
    }

} // namespace lfs::vis
