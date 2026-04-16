/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "sequencer/rml_sequencer_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "internal/resource_paths.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/render_constants.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <imgui_impl_opengl3.h>
#include <imgui.h>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;
        constexpr float PLAYHEAD_HIT_RADIUS = 6.0f;
        constexpr float PLAYHEAD_HANDLE_WIDTH = 8.0f;

        constexpr std::array<float, 5> SPEED_PRESETS = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};

        [[nodiscard]] size_t findSpeedIndex(const float speed) {
            size_t best = 2;
            float best_diff = 100.0f;
            for (size_t i = 0; i < SPEED_PRESETS.size(); ++i) {
                const float diff = std::abs(SPEED_PRESETS[i] - speed);
                if (diff < best_diff) {
                    best_diff = diff;
                    best = i;
                }
            }
            return best;
        }

        [[nodiscard]] std::string formatSpeed(const float speed) {
            if (speed >= 1.0f)
                return std::format("{}x", static_cast<int>(speed));
            return std::format("{:.2g}x", speed);
        }

        [[nodiscard]] std::string formatPresetShort(const lfs::io::video::VideoPreset preset) {
            return lfs::io::video::getPresetInfo(preset).name;
        }

        [[nodiscard]] std::string formatTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return std::format("{}:{:05.2f}", mins, secs);
        }

        [[nodiscard]] std::string formatTimeShort(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const int secs = static_cast<int>(seconds) % 60;
            if (mins > 0) {
                return std::format("{}:{:02d}", mins, secs);
            }
            return std::format("{}s", secs);
        }

        [[nodiscard]] bool hasSelectedKeyframe(const std::vector<sequencer::KeyframeId>& selected_keyframes,
                                               const sequencer::KeyframeId id) {
            return std::find(selected_keyframes.begin(), selected_keyframes.end(), id) !=
                   selected_keyframes.end();
        }

        void addSelectedKeyframe(std::vector<sequencer::KeyframeId>& selected_keyframes,
                                 const sequencer::KeyframeId id) {
            if (!hasSelectedKeyframe(selected_keyframes, id))
                selected_keyframes.push_back(id);
        }

        void removeSelectedKeyframe(std::vector<sequencer::KeyframeId>& selected_keyframes,
                                    const sequencer::KeyframeId id) {
            if (const auto it = std::find(selected_keyframes.begin(), selected_keyframes.end(), id);
                it != selected_keyframes.end()) {
                selected_keyframes.erase(it);
            }
        }

        [[nodiscard]] uint64_t selectedKeyframeSignature(std::vector<sequencer::KeyframeId> selected_keyframes) {
            std::sort(selected_keyframes.begin(), selected_keyframes.end());
            uint64_t signature = 1469598103934665603ull;
            for (const auto id : selected_keyframes) {
                signature ^= id;
                signature *= 1099511628211ull;
            }
            return signature;
        }

        [[nodiscard]] float clampCenteredSpan(const float center,
                                              const float extent,
                                              const float span) {
            if (extent <= 0.0f)
                return 0.0f;

            const float half_span = std::max(span * 0.5f, 0.0f);
            if (extent <= span)
                return extent * 0.5f;

            return std::clamp(center, half_span, extent - half_span);
        }

        void forwardFocusedKeyboardInput(Rml::Context* const context,
                                         const PanelInputState& input) {
            const int mods = gui::sdlModsToRml(input.key_ctrl, input.key_shift,
                                               input.key_alt, input.key_super);
            for (const int sc : input.keys_pressed) {
                const auto rml_key = gui::sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    context->ProcessKeyDown(rml_key, mods);
            }
            for (const int sc : input.keys_released) {
                const auto rml_key = gui::sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    context->ProcessKeyUp(rml_key, mods);
            }
        }

    } // namespace

    using gui::rml_theme::colorToRml;
    using gui::rml_theme::colorToRmlAlpha;
    using namespace panel_config;

    RmlSequencerPanel::RmlSequencerPanel(SequencerController& controller, gui::panels::SequencerUIState& ui_state,
                                         gui::RmlUIManager* rml_manager)
        : controller_(controller),
          ui_state_(ui_state),
          rml_manager_(rml_manager) {
        assert(rml_manager_);
        transport_listener_.panel = this;
        quality_scrub_listener_.panel = this;
    }

    RmlSequencerPanel::~RmlSequencerPanel() = default;

    void RmlSequencerPanel::TransportClickListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        auto* el = event.GetCurrentElement();
        if (!el)
            return;

        const auto& id = el->GetId();
        auto& ctrl = panel->controller_;
        auto& ui = panel->ui_state_;

        if (id == "btn-skip-back")
            ctrl.seekToFirstKeyframe();
        else if (id == "btn-prev-keyframe")
            ctrl.seekToPreviousKeyframe();
        else if (id == "btn-stop")
            ctrl.stop();
        else if (id == "btn-play")
            ctrl.togglePlayPause();
        else if (id == "btn-next-keyframe")
            ctrl.seekToNextKeyframe();
        else if (id == "btn-skip-forward")
            ctrl.seekToLastKeyframe();
        else if (id == "btn-loop") {
            ctrl.toggleLoop();
            lfs::core::events::state::KeyframeListChanged{.count = ctrl.timeline().realKeyframeCount()}.emit();
        } else if (id == "btn-add")
            lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
        else if (id == "btn-camera-path")
            ui.show_camera_path = !ui.show_camera_path;
        else if (id == "btn-snap")
            ui.snap_to_grid = !ui.snap_to_grid;
        else if (id == "btn-follow") {
            ui.follow_playback = !ui.follow_playback;
            if (ui.follow_playback)
                ui.show_pip_preview = false;
        } else if (id == "btn-film-strip")
            ui.show_film_strip = !ui.show_film_strip;
        else if (id == "btn-preview") {
            ui.show_pip_preview = !ui.show_pip_preview;
            if (ui.show_pip_preview)
                ui.follow_playback = false;
        } else if (id == "btn-equirect") {
            ui.equirectangular = !ui.equirectangular;
            auto event = lfs::core::events::ui::RenderSettingsChanged{};
            event.equirectangular = ui.equirectangular;
            event.emit();
        } else if (id == "btn-speed") {
            const size_t idx = findSpeedIndex(ui.playback_speed);
            const size_t next = (idx + 1) % SPEED_PRESETS.size();
            ui.playback_speed = SPEED_PRESETS[next];
            ctrl.setPlaybackSpeed(ui.playback_speed);
        } else if (id == "btn-format") {
            using lfs::io::video::VideoPreset;
            auto p = static_cast<int>(ui.preset);
            p = (p + 1) % static_cast<int>(VideoPreset::CUSTOM);
            ui.preset = static_cast<VideoPreset>(p);
            const auto info = lfs::io::video::getPresetInfo(ui.preset);
            ui.custom_width = info.width;
            ui.custom_height = info.height;
            ui.framerate = info.framerate;
        } else if (id == "btn-save-path")
            panel->save_path_requested_ = true;
        else if (id == "btn-load-path")
            panel->load_path_requested_ = true;
        else if (id == "btn-export")
            panel->export_requested_ = true;
        else if (id == "btn-dock-toggle")
            panel->dock_toggle_requested_ = true;
        else if (id == "btn-close-panel")
            panel->close_panel_requested_ = true;
        else if (id == "btn-clear") {
            float sx = panel->cached_panel_x_;
            float sy = panel->cached_panel_y_;
            auto abs_offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
            sx = panel->cached_panel_x_ + abs_offset.x;
            sy = panel->cached_panel_y_ + abs_offset.y + el->GetBox().GetSize().y;
            panel->transport_ctx_request_ = {TransportContextMenuRequest::Target::CLEAR, sx, sy};
        }
    }

    TimelineContextMenuState RmlSequencerPanel::consumeContextMenu() {
        TimelineContextMenuState state;
        if (context_menu_open_) {
            state.open = true;
            state.time = context_menu_time_;
            state.keyframe = context_menu_keyframe_;
            context_menu_open_ = false;
        }
        return state;
    }

    TransportContextMenuRequest RmlSequencerPanel::consumeTransportContextMenu() {
        auto req = transport_ctx_request_;
        transport_ctx_request_ = {};
        return req;
    }

    TimeEditRequest RmlSequencerPanel::consumeTimeEditRequest() {
        TimeEditRequest req;
        if (editing_keyframe_time_) {
            const auto& keyframes = controller_.timeline().keyframes();
            if (editing_keyframe_index_ < keyframes.size()) {
                req.active = true;
                req.keyframe_index = editing_keyframe_index_;
                req.current_time = keyframes[editing_keyframe_index_].time;
            }
            editing_keyframe_time_ = false;
        }
        return req;
    }

    FocalEditRequest RmlSequencerPanel::consumeFocalEditRequest() {
        FocalEditRequest req;
        if (editing_focal_length_) {
            req.active = true;
            req.keyframe_index = editing_focal_index_;
            req.current_focal_mm = std::stof(focal_edit_buffer_);
            editing_focal_length_ = false;
        }
        return req;
    }

    void RmlSequencerPanel::destroyGLResources() {
        clearPendingComposite();
        unregisterFilmStripSources();
        clearFilmThumbPool();
        if (el_film_strip_gaps_)
            el_film_strip_gaps_->SetInnerRML("");
        if (el_film_strip_markers_)
            el_film_strip_markers_->SetInnerRML("");
        if (el_film_strip_dividers_)
            el_film_strip_dividers_->SetInnerRML("");
        if (el_film_strip_sprockets_top_)
            el_film_strip_sprockets_top_->SetInnerRML("");
        if (el_film_strip_sprockets_bottom_)
            el_film_strip_sprockets_bottom_->SetInnerRML("");
        fbo_.destroy();
    }

    void RmlSequencerPanel::clearPendingComposite() {
        pending_foreground_composite_ = false;
        pending_composite_x_ = 0.0f;
        pending_composite_y_ = 0.0f;
        pending_composite_width_ = 0.0f;
        pending_composite_height_ = 0.0f;
    }

    void RmlSequencerPanel::compositeToScreen(const int screen_w, const int screen_h) {
        if (!pending_foreground_composite_ || !fbo_.valid() || screen_w <= 0 || screen_h <= 0) {
            clearPendingComposite();
            return;
        }

        ImDrawList draw_list(ImGui::GetDrawListSharedData());
        draw_list._ResetForNewFrame();
        draw_list.PushTextureID(ImGui::GetIO().Fonts->TexID);
        draw_list.PushClipRectFullScreen();
        gui::widgets::DrawFloatingWindowShadow(
            &draw_list,
            {pending_composite_x_, pending_composite_y_},
            {pending_composite_width_, pending_composite_height_},
            theme().sizes.window_rounding);
        draw_list.PopClipRect();

        if (!draw_list.CmdBuffer.empty() && !draw_list.VtxBuffer.empty()) {
            ImDrawData draw_data{};
            draw_data.DisplayPos = ImVec2(0.0f, 0.0f);
            draw_data.DisplaySize = ImVec2(static_cast<float>(screen_w),
                                           static_cast<float>(screen_h));
            draw_data.FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;
            draw_data.Valid = true;
            draw_data.AddDrawList(&draw_list);
            ImGui_ImplOpenGL3_RenderDrawData(&draw_data);
        }

        fbo_.blitToScreen(pending_composite_x_, pending_composite_y_,
                          pending_composite_width_, pending_composite_height_,
                          screen_w, screen_h);
        clearPendingComposite();
    }

    void RmlSequencerPanel::initContext(const int width, const int height) {
        if (rml_context_)
            return;

        cached_dp_ratio_ = rml_manager_->getDpRatio();
        rml_context_ = rml_manager_->createContext("sequencer", width, height);
        if (!rml_context_)
            return;

        try {
            const auto full_path = lfs::vis::getAssetPath("rmlui/sequencer.rml");
            document_ = rml_context_->LoadDocument(full_path.string());
            if (document_) {
                document_->Show();
                cacheElements();
            } else {
                LOG_ERROR("RmlUI: failed to load sequencer.rml");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("RmlUI: sequencer resource not found: {}", e.what());
        }
    }

    void RmlSequencerPanel::cacheElements() {
        assert(document_);
        el_panel_ = document_->GetElementById("panel");
        el_floating_header_ = document_->GetElementById("floating-header");
        el_ruler_ = document_->GetElementById("ruler");
        el_track_bar_ = document_->GetElementById("track-bar");
        el_keyframes_ = document_->GetElementById("keyframes");
        el_playhead_ = document_->GetElementById("playhead");
        el_hint_ = document_->GetElementById("hint");
        el_current_time_ = document_->GetElementById("current-time");
        el_duration_ = document_->GetElementById("duration");
        el_play_icon_ = document_->GetElementById("play-icon");
        el_btn_loop_ = document_->GetElementById("btn-loop");
        el_timeline_ = document_->GetElementById("timeline");
        el_header_ = document_->GetElementById("header");
        el_easing_stripe_ = document_->GetElementById("easing-stripe");
        el_easing_segments_ = document_->GetElementById("easing-segments");
        el_easing_curves_ = document_->GetElementById("easing-curves");
        el_easing_indicators_ = document_->GetElementById("easing-indicators");
        el_film_strip_panel_ = document_->GetElementById("film-strip-panel");
        el_film_strip_groove_ = document_->GetElementById("film-strip-groove");
        el_film_strip_gaps_ = document_->GetElementById("film-strip-gaps");
        el_film_strip_thumbs_ = document_->GetElementById("film-strip-thumbs");
        el_film_strip_markers_ = document_->GetElementById("film-strip-markers");
        el_film_strip_dividers_ = document_->GetElementById("film-strip-dividers");
        el_film_strip_sprockets_top_ = document_->GetElementById("film-strip-sprockets-top");
        el_film_strip_sprockets_bottom_ = document_->GetElementById("film-strip-sprockets-bottom");
        el_panel_guides_ = document_->GetElementById("panel-guides");
        el_guide_playhead_ = document_->GetElementById("guide-playhead");
        el_guide_selected_ = document_->GetElementById("guide-selected");
        el_guide_hovered_ = document_->GetElementById("guide-hovered");
        el_guide_strip_hover_ = document_->GetElementById("guide-strip-hover");
        el_timeline_tooltip_ = document_->GetElementById("timeline-tooltip");

        el_btn_camera_path_ = document_->GetElementById("btn-camera-path");
        el_btn_snap_ = document_->GetElementById("btn-snap");
        el_btn_follow_ = document_->GetElementById("btn-follow");
        el_btn_film_strip_ = document_->GetElementById("btn-film-strip");
        el_btn_preview_ = document_->GetElementById("btn-preview");
        el_speed_label_ = document_->GetElementById("speed-label");
        el_format_label_ = document_->GetElementById("format-label");
        el_resolution_info_ = document_->GetElementById("resolution-info");
        el_quality_scrub_ = document_->GetElementById("quality-scrub");
        el_quality_fill_ = document_->GetElementById("quality-fill");
        el_quality_display_ = document_->GetElementById("quality-display");
        el_quality_input_ = document_->GetElementById("quality-input");
        el_btn_equirect_ = document_->GetElementById("btn-equirect");
        el_btn_save_ = document_->GetElementById("btn-save-path");
        el_btn_load_ = document_->GetElementById("btn-load-path");
        el_btn_export_ = document_->GetElementById("btn-export");
        el_btn_clear_ = document_->GetElementById("btn-clear");
        el_transport_dock_sep_ = document_->GetElementById("dock-toggle-sep");
        el_btn_dock_toggle_ = document_->GetElementById("btn-dock-toggle");
        el_dock_toggle_label_ = document_->GetElementById("dock-toggle-label");
        el_btn_close_panel_ = document_->GetElementById("btn-close-panel");
        el_close_panel_label_ = document_->GetElementById("close-panel-label");

        elements_cached_ = el_ruler_ && el_keyframes_ && el_playhead_ &&
                           el_current_time_ && el_duration_ && el_play_icon_ &&
                           el_btn_loop_ && el_timeline_ && el_header_ &&
                           el_easing_stripe_ && el_easing_segments_ &&
                           el_easing_curves_ && el_easing_indicators_ &&
                           el_film_strip_panel_ && el_film_strip_groove_ &&
                           el_film_strip_gaps_ && el_film_strip_thumbs_ &&
                           el_film_strip_markers_ && el_film_strip_dividers_ &&
                           el_film_strip_sprockets_top_ && el_film_strip_sprockets_bottom_ &&
                           el_panel_guides_ && el_guide_playhead_ &&
                           el_guide_selected_ && el_guide_hovered_ &&
                           el_guide_strip_hover_ && el_timeline_tooltip_;
        if (!elements_cached_) {
            LOG_ERROR("RmlUI sequencer: missing DOM elements");
            return;
        }

        for (const char* btn_id : {"btn-skip-back", "btn-stop", "btn-play",
                                   "btn-prev-keyframe", "btn-next-keyframe", "btn-skip-forward",
                                   "btn-loop", "btn-add",
                                   "btn-camera-path", "btn-snap", "btn-follow",
                                   "btn-film-strip", "btn-preview", "btn-equirect", "btn-speed",
                                   "btn-format", "btn-save-path", "btn-load-path",
                                   "btn-export", "btn-clear", "btn-dock-toggle",
                                   "btn-close-panel"}) {
            auto* el = document_->GetElementById(btn_id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, &transport_listener_);
        }

        if (el_quality_scrub_) {
            el_quality_scrub_->AddEventListener(Rml::EventId::Mousedown, &quality_scrub_listener_);
            if (auto* body = document_->GetElementById("body")) {
                body->AddEventListener(Rml::EventId::Mousemove, &quality_scrub_listener_);
                body->AddEventListener(Rml::EventId::Mouseup, &quality_scrub_listener_);
            }
        }
        if (el_quality_input_) {
            el_quality_input_->AddEventListener(Rml::EventId::Change, &quality_scrub_listener_);
            el_quality_input_->AddEventListener(Rml::EventId::Blur, &quality_scrub_listener_);
        }
    }

    std::string RmlSequencerPanel::generateThemeRCSS(const lfs::vis::Theme& t) const {
        const auto& p = t.palette;

        const auto surface_alpha = colorToRml(p.surface);
        const auto border = colorToRmlAlpha(p.border, 0.4f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto text_dim_half = colorToRmlAlpha(p.text_dim, 0.5f);
        const auto bg_alpha = colorToRml(p.background);
        const auto border_dim = colorToRmlAlpha(p.border, 0.3f);
        const auto error = colorToRml(p.error);
        const auto primary_active = colorToRmlAlpha(p.primary, 0.20f);
        const auto primary_btn = colorToRmlAlpha(p.primary, 0.15f);
        const auto primary_btn_hover = colorToRmlAlpha(p.primary, 0.25f);
        const auto error_btn = colorToRmlAlpha(p.error, 0.15f);
        const auto error_btn_hover = colorToRmlAlpha(p.error, 0.30f);
        const auto primary_cam_bg = colorToRmlAlpha(p.primary, 0.30f);
        const auto primary_cam_border = colorToRmlAlpha(p.primary, 0.50f);
        const auto primary_export_border = colorToRmlAlpha(p.primary, 0.40f);
        const auto surface_bright_alpha = colorToRmlAlpha(p.surface_bright, 0.30f);
        const auto primary_color = colorToRml(p.primary);
        const auto primary_tint = colorToRmlAlpha(p.primary, 0.18f);
        const auto primary_fill = colorToRmlAlpha(p.primary, 0.25f);
        const auto primary_edge = colorToRmlAlpha(p.primary, 0.90f);
        const auto primary_outline = colorToRmlAlpha(p.primary, 0.65f);
        const auto secondary_color = colorToRml(p.secondary);
        const auto secondary_tint = colorToRmlAlpha(p.secondary, 0.14f);
        const auto secondary_fill = colorToRmlAlpha(p.secondary, 0.25f);
        const auto secondary_edge = colorToRmlAlpha(p.secondary, 0.70f);
        const auto secondary_outline = colorToRmlAlpha(p.secondary, 0.50f);
        const auto guide_hover = colorToRmlAlpha(p.secondary, 0.75f);
        const auto guide_strip_hover = colorToRmlAlpha(p.text_dim, 0.55f);
        const auto guide_selected = colorToRmlAlpha(p.primary, 0.85f);
        const auto guide_playhead = colorToRml(p.error);
        const auto film_strip_groove = colorToRml(p.background);
        const auto film_strip_gap = colorToRmlAlpha(p.surface, 0.30f);
        const auto film_strip_gap_stripe = colorToRmlAlpha(p.border, 0.18f);
        const auto film_thumb_midline_shadow = "rgba(0, 0, 0, 70)";
        const auto film_marker_shadow = "rgba(0, 0, 0, 78)";
        const auto divider_color = colorToRmlAlpha(p.text_dim, 0.15f);
        const auto sprocket_color = colorToRmlAlpha(p.text_dim, 0.30f);
        const auto tooltip_surface = colorToRmlAlpha(p.surface, 0.96f);
        const auto tooltip_border = colorToRmlAlpha(p.border, 0.75f);
        const auto tooltip_text_dim = colorToRmlAlpha(p.text_dim, 0.95f);
        const int rounding = static_cast<int>(t.sizes.window_rounding);

        const std::string radius_str = film_strip_attached_
                                           ? std::format("{}dp {}dp 0dp 0dp", rounding, rounding)
                                           : std::format("{}dp", rounding);

        std::string css = std::format(
            "#panel {{ background-color: {}; border-width: 1dp; border-color: {}; "
            "border-radius: {}; }}\n"
            ".transport-icon {{ image-color: {}; }}\n"
            "#track-bar {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#hint {{ color: {}; }}\n"
            ".ruler-tick.major {{ background-color: {}; }}\n"
            ".ruler-tick.minor {{ background-color: {}; }}\n"
            ".ruler-label {{ color: {}; }}\n"
            "#playhead-handle {{ background-color: {}; }}\n"
            "#current-time {{ color: {}; }}\n"
            "#duration {{ color: {}; }}\n"
            "#easing-stripe {{ background-color: {}; border-top: 1dp {}; }}\n"
            "#transport-row {{ border-bottom: 1dp {}; }}\n"
            ".transport-sep {{ background-color: {}; }}\n"
            ".transport-label {{ color: {}; }}\n"
            ".transport-info {{ color: {}; }}\n"
            ".transport-btn.toggle.active {{ background-color: {}; }}\n"
            ".transport-btn.primary {{ background-color: {}; }}\n"
            ".transport-btn.primary:hover {{ background-color: {}; }}\n"
            ".transport-btn.error:hover {{ background-color: {}; }}\n"
            "#btn-camera-path.active {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#btn-add .transport-icon {{ image-color: {}; }}\n"
            ".speed-val {{ color: {}; }}\n"
            ".speed-text {{ color: {}; }}\n"
            ".dropdown-arrow {{ color: {}; }}\n"
            ".snap-check {{ border-color: {}; }}\n"
            "#btn-snap.active .snap-check {{ background-color: {}; border-color: {}; }}\n"
            ".format-badge {{ background-color: {}; }}\n"
            "#btn-export {{ border-width: 1dp; border-color: {}; }}\n"
            "#btn-clear {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#btn-clear .transport-icon {{ image-color: {}; }}\n"
            ".ctx-indicator {{ color: {}; }}\n",
            surface_alpha, border, radius_str,
            text,
            bg_alpha, border_dim,
            text_dim_half,
            text_dim,
            text_dim_half,
            text_dim,
            error,
            text,
            text_dim,
            surface_alpha, border_dim,
            border_dim,
            border_dim,
            text,
            text_dim,
            primary_active,
            primary_btn,
            primary_btn_hover,
            error_btn_hover,
            primary_cam_bg, primary_cam_border,
            error,
            text,
            text_dim,
            text_dim,
            text_dim,
            primary_color, primary_color,
            surface_bright_alpha,
            primary_export_border,
            error_btn, error_btn,
            error,
            text_dim_half);

        css += std::format(
            "#film-strip-panel {{ background-color: {}; border-left: 1dp {}; border-right: 1dp {}; border-bottom: 1dp {}; border-radius: 0dp 0dp {}dp {}dp; }}\n"
            "#film-strip-groove {{ background-color: {}; }}\n"
            ".film-strip-gap {{ background-color: {}; }}\n"
            ".film-strip-gap-stripe {{ background-color: {}; }}\n"
            ".film-strip-divider {{ background-color: {}; }}\n"
            ".film-strip-sprocket {{ background-color: {}; }}\n"
            ".film-thumb-tint.hovered-keyframe {{ background-color: {}; }}\n"
            ".film-thumb-tint.selected {{ background-color: {}; }}\n"
            ".film-thumb.contains-hovered-keyframe .film-thumb-edge {{ background-color: {}; }}\n"
            ".film-thumb.contains-selected .film-thumb-edge {{ background-color: {}; }}\n"
            ".film-thumb.hovered .film-thumb-outline {{ border-color: {}; }}\n"
            ".film-thumb.contains-hovered-keyframe .film-thumb-outline {{ border-color: {}; }}\n"
            ".film-thumb.contains-selected .film-thumb-outline {{ border-color: {}; }}\n"
            ".film-thumb-midline.shadow {{ background-color: {}; }}\n"
            ".film-thumb-midline.main {{ background-color: {}; }}\n"
            ".film-thumb.contains-hovered-keyframe .film-thumb-midline.main {{ background-color: {}; }}\n"
            ".film-thumb.contains-selected .film-thumb-midline.main {{ background-color: {}; }}\n"
            ".film-thumb.hovered .film-thumb-midline.main {{ background-color: {}; }}\n"
            ".film-strip-marker-line.shadow {{ background-color: {}; }}\n"
            ".film-strip-marker-line.main {{ background-color: {}; }}\n"
            ".film-strip-marker.hovered .film-strip-marker-line.main, .film-strip-marker.hovered .film-strip-marker-cap {{ background-color: {}; }}\n"
            ".film-strip-marker.selected .film-strip-marker-line.main, .film-strip-marker.selected .film-strip-marker-cap {{ background-color: {}; }}\n"
            ".film-strip-marker-cap {{ background-color: {}; }}\n"
            ".easing-segment.primary {{ background-color: {}; }}\n"
            ".easing-segment.secondary {{ background-color: {}; }}\n"
            ".easing-curve-segment {{ background-color: {}; }}\n"
            ".easing-dot.primary, .easing-indicator.primary.ease-in-out {{ background-color: {}; }}\n"
            ".easing-dot.secondary, .easing-indicator.secondary.ease-in-out {{ background-color: {}; }}\n"
            ".easing-indicator.primary.ease-in {{ border-bottom-color: {}; }}\n"
            ".easing-indicator.secondary.ease-in {{ border-bottom-color: {}; }}\n"
            ".easing-indicator.primary.ease-out {{ border-top-color: {}; }}\n"
            ".easing-indicator.secondary.ease-out {{ border-top-color: {}; }}\n"
            ".timeline-guide.hovered {{ background-color: {}; }}\n"
            ".timeline-guide.selected {{ background-color: {}; }}\n"
            ".timeline-guide.playhead {{ background-color: {}; }}\n"
            ".timeline-guide.strip-hover {{ background-color: {}; }}\n"
            "#timeline-tooltip {{ background-color: {}; border-color: {}; }}\n"
            ".timeline-tooltip-line.title {{ color: {}; }}\n"
            ".timeline-tooltip-line {{ color: {}; }}\n",
            surface_alpha, border, border, border, rounding, rounding,
            film_strip_groove,
            film_strip_gap,
            film_strip_gap_stripe,
            divider_color,
            sprocket_color,
            secondary_tint,
            primary_tint,
            secondary_edge,
            primary_edge,
            text,
            secondary_outline,
            primary_outline,
            film_thumb_midline_shadow,
            text_dim_half,
            secondary_color,
            primary_color,
            text,
            film_marker_shadow,
            text_dim_half,
            secondary_color,
            primary_color,
            text_dim_half,
            primary_fill,
            secondary_fill,
            colorToRmlAlpha(p.primary, 0.50f),
            primary_color,
            secondary_color,
            primary_color,
            secondary_color,
            primary_color,
            secondary_color,
            guide_hover,
            guide_selected,
            guide_playhead,
            guide_strip_hover,
            tooltip_surface, tooltip_border,
            text,
            tooltip_text_dim);

        return css;
    }

    void RmlSequencerPanel::syncTheme() {
        if (!document_)
            return;

        const auto& p = lfs::vis::theme().palette;
        const bool layout_changed = film_strip_attached_ != last_film_strip_attached_ ||
                                    floating_ != last_floating_;
        if (!layout_changed && std::memcmp(last_synced_text_, &p.text, sizeof(last_synced_text_)) == 0)
            return;
        std::memcpy(last_synced_text_, &p.text, sizeof(last_synced_text_));
        last_film_strip_attached_ = film_strip_attached_;
        last_floating_ = floating_;

        if (base_rcss_.empty())
            base_rcss_ = gui::rml_theme::loadBaseRCSS("rmlui/sequencer.rcss");

        gui::rml_theme::applyTheme(document_, base_rcss_, gui::rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
    }

    void RmlSequencerPanel::updateButtonStates() {
        if (!elements_cached_)
            return;

        const bool playing = controller_.isPlaying();
        el_play_icon_->SetAttribute("src",
                                    playing ? "../icon/sequencer/pause.png"
                                            : "../icon/sequencer/play.png");

        auto* btn_play = document_->GetElementById("btn-play");
        if (btn_play)
            btn_play->SetAttribute("data-tooltip",
                                   playing ? "tooltip.seq_pause" : "tooltip.seq_play");

        const bool looping = controller_.loopMode() != LoopMode::ONCE;
        el_btn_loop_->SetClass("active", looping);
        el_btn_loop_->SetAttribute("data-tooltip",
                                   looping ? "tooltip.seq_loop_on" : "tooltip.seq_loop_off");
    }

    void RmlSequencerPanel::updatePlayhead() {
        if (!elements_cached_)
            return;

        const float tl_width = timelineWidth();
        if (tl_width <= 0.0f)
            return;

        const float x = clampCenteredSpan(
            timeToX(controller_.playhead(), 0.0f, tl_width),
            tl_width,
            PLAYHEAD_HANDLE_WIDTH * cached_dp_ratio_);
        el_playhead_->SetProperty("left", std::format("{:.1f}px", x));
    }

    void RmlSequencerPanel::updateTimeDisplay() {
        if (!elements_cached_)
            return;

        el_current_time_->SetInnerRML(formatTime(controller_.playhead()));

        const float end = controller_.timeline().empty()
                              ? sequencer_ui::DEFAULT_TIMELINE_DURATION
                              : controller_.timeline().endTime();
        el_duration_->SetInnerRML(" / " + formatTime(end));
    }

    void RmlSequencerPanel::rebuildKeyframes() {
        if (!elements_cached_)
            return;

        const auto& timeline = controller_.timeline();
        const auto& keyframes = timeline.keyframes();
        const size_t count = keyframes.size();

        std::erase_if(selected_keyframes_, [&timeline](const sequencer::KeyframeId id) {
            return !timeline.findKeyframeIndex(id).has_value();
        });

        const float timeline_width = timelineWidth();
        const uint64_t timeline_revision = controller_.timelineRevision();
        const uint64_t selection_revision = controller_.selectionRevision();
        const uint64_t selected_keyframes_signature = selectedKeyframeSignature(selected_keyframes_);

        if (!dragging_keyframe_ &&
            count == last_keyframe_count_ &&
            zoom_level_ == last_zoom_level_ &&
            pan_offset_ == last_pan_offset_ &&
            timeline_revision == last_timeline_revision_ &&
            selection_revision == last_selection_revision_ &&
            selected_keyframes_signature == last_selected_keyframes_signature_ &&
            timeline_width == last_kf_width_) {
            return;
        }
        last_keyframe_count_ = count;
        last_zoom_level_ = zoom_level_;
        last_pan_offset_ = pan_offset_;
        last_kf_width_ = timeline_width;
        last_timeline_revision_ = timeline_revision;
        last_selection_revision_ = selection_revision;
        last_selected_keyframes_signature_ = selected_keyframes_signature;
        if (timeline_width <= 0.0f)
            return;

        const auto& p = lfs::vis::theme().palette;

        if (count == 0) {
            while (!keyframe_elements_.empty()) {
                el_keyframes_->RemoveChild(keyframe_elements_.back());
                keyframe_elements_.pop_back();
            }
            if (el_hint_)
                el_hint_->SetInnerRML(LOC(lichtfeld::Strings::Sequencer::EMPTY_HINT));
            return;
        }

        if (el_hint_)
            el_hint_->SetInnerRML("");

        while (keyframe_elements_.size() < count) {
            auto new_elem = document_->CreateElement("div");
            assert(new_elem);
            Rml::Element* raw = new_elem.get();
            el_keyframes_->AppendChild(std::move(new_elem));
            keyframe_elements_.push_back(raw);
        }
        while (keyframe_elements_.size() > count) {
            el_keyframes_->RemoveChild(keyframe_elements_.back());
            keyframe_elements_.pop_back();
        }

        for (size_t i = 0; i < count; ++i) {
            auto* el = keyframe_elements_[i];
            const float x = timeToX(keyframes[i].time, 0.0f, timeline_width);
            const bool selected = controller_.selectedKeyframe() == i ||
                                  hasSelectedKeyframe(selected_keyframes_, keyframes[i].id);
            const bool is_loop = keyframes[i].is_loop_point;

            const auto base = is_loop ? p.info : (i % 2 == 0 ? p.primary : p.secondary);
            auto fill = base;
            if (selected)
                fill = lighten(base, 0.2f);

            el->SetClassNames("keyframe");
            el->SetClass("loop-point", is_loop);
            el->SetClass("selected", selected);
            el->SetProperty("left", std::format("{:.1f}px", x));
            el->SetProperty("background-color", colorToRml(fill));
            el->SetProperty("border-color", selected ? colorToRml(p.text) : colorToRml(fill));
        }
    }

    void RmlSequencerPanel::rebuildRuler() {
        if (!elements_cached_)
            return;

        const float timeline_width = timelineWidth();
        const float display_end_time = getDisplayEndTime();

        if (zoom_level_ == last_ruler_zoom_ &&
            pan_offset_ == last_ruler_pan_ &&
            timeline_width == last_ruler_width_ &&
            display_end_time == last_ruler_display_end_)
            return;
        last_ruler_zoom_ = zoom_level_;
        last_ruler_pan_ = pan_offset_;
        last_ruler_width_ = timeline_width;
        last_ruler_display_end_ = display_end_time;
        if (timeline_width <= 0.0f)
            return;

        const float visible_duration = display_end_time;
        const float visible_start = pan_offset_;
        const float visible_end = visible_start + visible_duration;

        float major_interval = 1.0f;
        if (visible_duration > 60.0f)
            major_interval = 10.0f;
        else if (visible_duration > 30.0f)
            major_interval = 5.0f;
        else if (visible_duration > 10.0f)
            major_interval = 2.0f;
        else if (visible_duration <= 2.0f)
            major_interval = 0.5f;

        major_interval /= zoom_level_;
        const float minor_interval = major_interval / 4.0f;

        std::string html;
        html.reserve(2048);

        const float label_margin = 30.0f * cached_dp_ratio_;

        const float first_tick = std::floor(visible_start / minor_interval) * minor_interval;
        for (float t_val = first_tick; t_val <= visible_end + minor_interval * 0.5f; t_val += minor_interval) {
            if (t_val < 0.0f)
                continue;

            const float x = timeToX(t_val, 0.0f, timeline_width);
            if (x < 0.0f || x > timeline_width)
                continue;

            const float major_phase = std::fmod(t_val, major_interval);
            const bool is_major = major_phase < 0.01f || (major_interval - major_phase) < 0.01f;

            if (is_major) {
                html += std::format(
                    "<div class=\"ruler-tick major\" style=\"left: {:.1f}px;\" />", x);
                if (x + label_margin <= timeline_width) {
                    html += std::format(
                        "<span class=\"ruler-label\" style=\"left: {:.1f}px;\">{}</span>",
                        x + 4.0f * cached_dp_ratio_, formatTimeShort(t_val));
                }
            } else {
                html += std::format(
                    "<div class=\"ruler-tick minor\" style=\"left: {:.1f}px;\" />",
                    x);
            }
        }

        el_ruler_->SetInnerRML(html);
    }

    void RmlSequencerPanel::rebuildEasingStripe(const float timeline_x, const float timeline_width) {
        if (!elements_cached_)
            return;

        const auto& keyframes = controller_.timeline().keyframes();
        if (timeline_width <= 0.0f || keyframes.empty()) {
            el_easing_segments_->SetInnerRML("");
            el_easing_curves_->SetInnerRML("");
            el_easing_indicators_->SetInnerRML("");
            return;
        }

        constexpr int CURVE_SAMPLES = 20;
        const float stripe_h = EASING_STRIPE_HEIGHT * cached_dp_ratio_;
        const float y_center = stripe_h * 0.5f;
        const float amplitude = stripe_h * 0.35f;
        const float display_end = getDisplayEndTime();
        const float pan = pan_offset_;

        const auto localTimeToX = [&](const float time) -> float {
            return sequencer_ui::timeToScreenX(time, timeline_x, timeline_width, display_end, pan) - timeline_x;
        };

        std::string segments_html;
        std::string curves_html;
        std::string indicators_html;
        segments_html.reserve(512);
        curves_html.reserve(4096);
        indicators_html.reserve(1024);

        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const float x0 = localTimeToX(keyframes[i].time);
            const float x1 = localTimeToX(keyframes[i + 1].time);
            if (x1 <= x0)
                continue;

            segments_html += std::format(
                "<div class=\"easing-segment {}\" style=\"left:{:.1f}px;width:{:.1f}px;\"></div>",
                (i % 2 == 0) ? "primary" : "secondary",
                x0, x1 - x0);

            const auto easing = keyframes[i].easing;
            if (easing == sequencer::EasingType::LINEAR) {
                const float len = x1 - x0;
                curves_html += std::format(
                    "<div class=\"easing-curve-segment\" style=\"left:{:.1f}px;top:{:.1f}px;width:{:.1f}px;transform:rotate(0deg);\"></div>",
                    x0, y_center, len);
                continue;
            }

            for (int s = 0; s < CURVE_SAMPLES; ++s) {
                const float t0 = static_cast<float>(s) / static_cast<float>(CURVE_SAMPLES);
                const float t1 = static_cast<float>(s + 1) / static_cast<float>(CURVE_SAMPLES);
                const float eased0 = sequencer::applyEasing(t0, easing);
                const float eased1 = sequencer::applyEasing(t1, easing);
                const float px0 = x0 + t0 * (x1 - x0);
                const float px1 = x0 + t1 * (x1 - x0);
                const float py0 = y_center - (eased0 - t0) * amplitude;
                const float py1 = y_center - (eased1 - t1) * amplitude;
                const float dx = px1 - px0;
                const float dy = py1 - py0;
                const float len = std::sqrt(dx * dx + dy * dy);
                if (len < 0.25f)
                    continue;

                const float angle_deg = std::atan2(dy, dx) * 57.2957795f;
                curves_html += std::format(
                    "<div class=\"easing-curve-segment\" style=\"left:{:.1f}px;top:{:.1f}px;width:{:.1f}px;transform:rotate({:.2f}deg);\"></div>",
                    px0, py0, len, angle_deg);
            }
        }

        for (size_t i = 0; i < keyframes.size(); ++i) {
            const float kx = localTimeToX(keyframes[i].time);
            const char* tone = (i % 2 == 0) ? "primary" : "secondary";
            indicators_html += std::format(
                "<div class=\"easing-dot {}\" style=\"left:{:.1f}px;top:{:.1f}px;\"></div>",
                tone, kx, y_center);

            const auto easing = keyframes[i].easing;
            if (easing == sequencer::EasingType::LINEAR)
                continue;

            const float iy = y_center - stripe_h * 0.3f;
            const char* easing_class = "";
            switch (easing) {
            case sequencer::EasingType::EASE_IN: easing_class = "ease-in"; break;
            case sequencer::EasingType::EASE_OUT: easing_class = "ease-out"; break;
            case sequencer::EasingType::EASE_IN_OUT: easing_class = "ease-in-out"; break;
            default: break;
            }

            indicators_html += std::format(
                "<div class=\"easing-indicator {} {}\" style=\"left:{:.1f}px;top:{:.1f}px;\"></div>",
                easing_class, tone, kx, iy);
        }

        el_easing_segments_->SetInnerRML(segments_html);
        el_easing_curves_->SetInnerRML(curves_html);
        el_easing_indicators_->SetInnerRML(indicators_html);
    }

    void RmlSequencerPanel::ensureFilmThumbPool(const size_t count) {
        if (!elements_cached_ || !el_film_strip_thumbs_)
            return;

        while (film_thumb_elements_.size() < count) {
            auto thumb = document_->CreateElement("div");
            auto* thumb_raw = thumb.get();
            thumb_raw->SetClassNames("film-thumb");

            auto image = document_->CreateElement("img");
            image->SetClassNames("film-thumb-image");
            thumb_raw->AppendChild(std::move(image));

            auto tint_hover = document_->CreateElement("div");
            tint_hover->SetClassNames("film-thumb-tint hovered-keyframe");
            thumb_raw->AppendChild(std::move(tint_hover));

            auto tint_selected = document_->CreateElement("div");
            tint_selected->SetClassNames("film-thumb-tint selected");
            thumb_raw->AppendChild(std::move(tint_selected));

            auto edge_top = document_->CreateElement("div");
            edge_top->SetClassNames("film-thumb-edge top");
            thumb_raw->AppendChild(std::move(edge_top));

            auto edge_bottom = document_->CreateElement("div");
            edge_bottom->SetClassNames("film-thumb-edge bottom");
            thumb_raw->AppendChild(std::move(edge_bottom));

            auto outline = document_->CreateElement("div");
            outline->SetClassNames("film-thumb-outline");
            thumb_raw->AppendChild(std::move(outline));

            auto mid_shadow = document_->CreateElement("div");
            mid_shadow->SetClassNames("film-thumb-midline shadow");
            thumb_raw->AppendChild(std::move(mid_shadow));

            auto mid_main = document_->CreateElement("div");
            mid_main->SetClassNames("film-thumb-midline main");
            thumb_raw->AppendChild(std::move(mid_main));

            el_film_strip_thumbs_->AppendChild(std::move(thumb));
            film_thumb_elements_.push_back(thumb_raw);
        }
    }

    void RmlSequencerPanel::clearFilmThumbPool() {
        if (!el_film_strip_thumbs_)
            return;

        while (!film_thumb_elements_.empty()) {
            auto* el = film_thumb_elements_.back();
            if (el && el->GetNumChildren() > 0) {
                if (auto* image = el->GetChild(0))
                    image->SetAttribute("src", "");
            }
            el_film_strip_thumbs_->RemoveChild(el);
            film_thumb_elements_.pop_back();
        }
    }

    void RmlSequencerPanel::unregisterFilmStripSources() {
        auto* render = rml_manager_ ? rml_manager_->getRenderInterface() : nullptr;
        if (!render) {
            registered_film_strip_sources_.clear();
            return;
        }

        for (const auto& source : registered_film_strip_sources_)
            render->unregister_external_texture(source);
        registered_film_strip_sources_.clear();
    }

    void RmlSequencerPanel::rebuildFilmStripDecor(const float timeline_width) {
        if (!elements_cached_)
            return;

        const float thumb_display_h = gui::FilmStripRenderer::STRIP_HEIGHT -
                                      gui::FilmStripRenderer::THUMB_PADDING * 2.0f;
        const float base_thumb_w = thumb_display_h * (static_cast<float>(gui::FilmStripRenderer::THUMB_WIDTH) /
                                                      static_cast<float>(gui::FilmStripRenderer::THUMB_HEIGHT));
        const int num_thumbs = sequencer_ui::thumbnailCount(timeline_width, base_thumb_w, zoom_level_);
        const float actual_thumb_w = num_thumbs > 0 ? timeline_width / static_cast<float>(num_thumbs) : 0.0f;
        const float groove_w = timeline_width + gui::FilmStripRenderer::THUMB_PADDING * 2.0f;

        std::string divider_html;
        divider_html.reserve(256);
        for (int i = 1; i < num_thumbs; ++i) {
            divider_html += std::format(
                "<div class=\"film-strip-divider\" style=\"left:{:.1f}px;\"></div>",
                gui::FilmStripRenderer::THUMB_PADDING + actual_thumb_w * static_cast<float>(i));
        }
        el_film_strip_dividers_->SetInnerRML(divider_html);

        std::string sprocket_top_html;
        std::string sprocket_bottom_html;
        const float sprocket_start = gui::FilmStripRenderer::SPROCKET_SPACING * 0.5f;
        const int sprocket_count = static_cast<int>(groove_w / gui::FilmStripRenderer::SPROCKET_SPACING);
        sprocket_top_html.reserve(static_cast<size_t>(sprocket_count) * 48);
        sprocket_bottom_html.reserve(static_cast<size_t>(sprocket_count) * 48);
        for (int i = 0; i < sprocket_count; ++i) {
            const float cx = sprocket_start + static_cast<float>(i) * gui::FilmStripRenderer::SPROCKET_SPACING;
            const float sx = cx - gui::FilmStripRenderer::SPROCKET_W * 0.5f;
            sprocket_top_html += std::format(
                "<div class=\"film-strip-sprocket top\" style=\"left:{:.1f}px;\"></div>", sx);
            sprocket_bottom_html += std::format(
                "<div class=\"film-strip-sprocket bottom\" style=\"left:{:.1f}px;\"></div>", sx);
        }
        el_film_strip_sprockets_top_->SetInnerRML(sprocket_top_html);
        el_film_strip_sprockets_bottom_->SetInnerRML(sprocket_bottom_html);
    }

    void RmlSequencerPanel::rebuildFilmStrip(float timeline_x, const float timeline_width,
                                             const float strip_y, const PanelInputState& input,
                                             RenderingManager* rm, SceneManager* sm,
                                             gui::FilmStripRenderer& film_strip) {
        if (!elements_cached_)
            return;

        if (!film_strip_attached_) {
            if (film_strip_scrubbing_) {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
            unregisterFilmStripSources();
            clearFilmThumbPool();
            el_film_strip_gaps_->SetInnerRML("");
            el_film_strip_markers_->SetInnerRML("");
            el_film_strip_dividers_->SetInnerRML("");
            el_film_strip_sprockets_top_->SetInnerRML("");
            el_film_strip_sprockets_bottom_->SetInnerRML("");
            updateTimelineTooltip(film_strip, input);
            return;
        }

        std::optional<float> selected_keyframe_time;
        if (const auto selected = controller_.selectedKeyframe(); selected.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframe(*selected))
                selected_keyframe_time = keyframe->time;
        }

        std::optional<float> hovered_keyframe_time;
        if (const auto hovered_id = hoveredKeyframeId(); hovered_id.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframeById(*hovered_id))
                hovered_keyframe_time = keyframe->time;
        }

        gui::FilmStripRenderer::RenderOptions options;
        options.panel_x = cached_panel_x_;
        options.panel_width = cached_panel_width_;
        options.timeline_x = timeline_x;
        options.timeline_width = timeline_width;
        options.strip_y = strip_y;
        options.mouse_x = input.mouse_x;
        options.mouse_y = input.mouse_y;
        options.zoom_level = zoom_level_;
        options.pan_offset = pan_offset_;
        options.display_end_time = getDisplayEndTime();
        options.selected_keyframe_id = controller_.selectedKeyframeId();
        options.hovered_keyframe_id = hoveredKeyframeId();
        options.selected_keyframe_time = selected_keyframe_time;
        options.hovered_keyframe_time = hovered_keyframe_time;
        film_strip.render(controller_, rm, sm, options);

        handleFilmStripInteraction(timeline_x, timeline_width, input, film_strip);
        rebuildFilmStripDecor(timeline_width);

        const float groove_origin_x = timeline_x - gui::FilmStripRenderer::THUMB_PADDING;

        std::string gaps_html;
        if (controller_.timeline().size() >= 2) {
            const float visible_left_x = gui::FilmStripRenderer::THUMB_PADDING;
            const float visible_right_x = gui::FilmStripRenderer::THUMB_PADDING + timeline_width;
            const float anim_start_x = std::clamp(
                timeToX(controller_.timeline().startTime(), timeline_x, timeline_width) - timeline_x +
                    gui::FilmStripRenderer::THUMB_PADDING,
                visible_left_x, visible_right_x);
            const float anim_end_x = std::clamp(
                timeToX(controller_.timeline().endTime(), timeline_x, timeline_width) - timeline_x +
                    gui::FilmStripRenderer::THUMB_PADDING,
                visible_left_x, visible_right_x);

            const auto append_gap_region = [&](const float x_min, const float x_max) {
                if (x_max <= x_min)
                    return;

                gaps_html += std::format(
                    "<div class=\"film-strip-gap\" style=\"left:{:.1f}px;width:{:.1f}px;\">",
                    x_min, x_max - x_min);
                const float stripe_span = gui::FilmStripRenderer::STRIP_HEIGHT -
                                          gui::FilmStripRenderer::THUMB_PADDING * 2.0f;
                for (float stripe_x = -stripe_span; stripe_x < (x_max - x_min) + stripe_span;
                     stripe_x += 10.0f) {
                    gaps_html += std::format(
                        "<div class=\"film-strip-gap-stripe\" style=\"left:{:.1f}px;top:{:.1f}px;height:{:.1f}px;transform:rotate(45deg);\"></div>",
                        stripe_x, stripe_span, stripe_span * 1.4142f);
                }
                gaps_html += "</div>";
            };

            if (anim_start_x > visible_left_x)
                append_gap_region(visible_left_x, anim_start_x);
            if (anim_end_x < visible_right_x)
                append_gap_region(anim_end_x, visible_right_x);
        }
        el_film_strip_gaps_->SetInnerRML(gaps_html);

        ensureFilmThumbPool(film_strip.thumbs().size());
        auto* render = rml_manager_ ? rml_manager_->getRenderInterface() : nullptr;
        if (!render)
            unregisterFilmStripSources();
        std::set<std::string> active_sources;
        for (size_t i = 0; i < film_thumb_elements_.size(); ++i) {
            auto* thumb_el = film_thumb_elements_[i];
            auto* image_el = thumb_el && thumb_el->GetNumChildren() > 0 ? thumb_el->GetChild(0) : nullptr;
            if (!thumb_el || !image_el)
                continue;

            if (i >= film_strip.thumbs().size()) {
                thumb_el->SetProperty("display", "none");
                image_el->SetAttribute("src", "");
                continue;
            }

            const auto& thumb = film_strip.thumbs()[i];
            const unsigned int texture_id = film_strip.textureIdForSlot(thumb.slot_idx);
            if (texture_id == 0) {
                thumb_el->SetProperty("display", "none");
                image_el->SetAttribute("src", "");
                continue;
            }

            const std::string source =
                std::format("sequencer-film-slot://{}-{}", thumb.slot_idx, texture_id);
            if (render)
                render->register_external_texture(source, texture_id,
                                                  gui::FilmStripRenderer::THUMB_WIDTH,
                                                  gui::FilmStripRenderer::THUMB_HEIGHT,
                                                  true);
            active_sources.insert(source);

            thumb_el->SetProperty("display", "block");
            thumb_el->SetProperty("left", std::format("{:.1f}px", thumb.screen_x - groove_origin_x));
            thumb_el->SetProperty("width", std::format("{:.1f}px", thumb.screen_width));
            thumb_el->SetClassNames("film-thumb");
            thumb_el->SetClass("hovered", thumb.hovered);
            thumb_el->SetClass("contains-selected", thumb.contains_selected);
            thumb_el->SetClass("contains-hovered-keyframe", thumb.contains_hovered_keyframe);
            thumb_el->SetClass("stale", thumb.stale);

            const auto current_source = image_el->GetAttribute<Rml::String>("src", "");
            if (current_source != source)
                image_el->SetAttribute("src", source);
        }

        for (auto it = registered_film_strip_sources_.begin(); it != registered_film_strip_sources_.end();) {
            if (!active_sources.contains(*it) && render)
                render->unregister_external_texture(*it);
            if (!active_sources.contains(*it))
                it = registered_film_strip_sources_.erase(it);
            else
                ++it;
        }
        registered_film_strip_sources_.insert(active_sources.begin(), active_sources.end());

        std::string markers_html;
        markers_html.reserve(film_strip.markers().size() * 196);
        for (const auto& marker : film_strip.markers()) {
            markers_html += std::format(
                "<div class=\"film-strip-marker{}{}\" style=\"left:{:.1f}px;\">"
                "<div class=\"film-strip-marker-line shadow\"></div>"
                "<div class=\"film-strip-marker-line main\"></div>"
                "<div class=\"film-strip-marker-cap top\"></div>"
                "<div class=\"film-strip-marker-cap bottom\"></div>"
                "</div>",
                marker.selected ? " selected" : "",
                marker.hovered ? " hovered" : "",
                marker.screen_x - groove_origin_x);
        }
        el_film_strip_markers_->SetInnerRML(markers_html);

        updateTimelineTooltip(film_strip, input);
    }

    void RmlSequencerPanel::updateTimelineGuides(const float timeline_x, const float timeline_width,
                                                 const gui::FilmStripRenderer& film_strip) {
        if (!elements_cached_ || timeline_width <= 0.0f)
            return;

        struct ElementBounds {
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
        };

        const auto document_offset = document_->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto measure = [document_offset](Rml::Element* const el) -> std::optional<ElementBounds> {
            if (!el)
                return std::nullopt;

            const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = el->GetBox().GetSize(Rml::BoxArea::Border);
            return ElementBounds{
                .x = offset.x - document_offset.x,
                .y = offset.y - document_offset.y,
                .width = size.x,
                .height = size.y,
            };
        };

        const auto timeline_bounds = measure(el_timeline_);
        const auto easing_bounds = measure(el_easing_stripe_);
        if (!timeline_bounds.has_value() || !easing_bounds.has_value())
            return;

        float guide_left = timeline_bounds->x;
        float guide_top = timeline_bounds->y;
        float guide_width = timeline_bounds->width;
        float guide_bottom = std::max(timeline_bounds->y + timeline_bounds->height,
                                      easing_bounds->y + easing_bounds->height);

        if (film_strip_attached_) {
            if (const auto film_strip_bounds = measure(el_film_strip_panel_); film_strip_bounds.has_value()) {
                guide_bottom = std::max(guide_bottom,
                                        film_strip_bounds->y + film_strip_bounds->height);
            }
        }

        guide_width = std::max(guide_width, 0.0f);
        el_panel_guides_->SetProperty("left", std::format("{:.1f}px", guide_left));
        el_panel_guides_->SetProperty("top", std::format("{:.1f}px", guide_top));
        el_panel_guides_->SetProperty("width", std::format("{:.1f}px", guide_width));
        el_panel_guides_->SetProperty("height", std::format("{:.1f}px", std::max(0.0f, guide_bottom - guide_top)));

        const auto set_guide = [guide_width](Rml::Element* const el,
                                             const std::optional<float> x,
                                             const float width_px = 1.0f) {
            if (!el)
                return;
            if (!x.has_value()) {
                el->SetProperty("display", "none");
                return;
            }
            const float clamped_center = clampCenteredSpan(*x, guide_width, width_px);
            el->SetProperty("display", "block");
            el->SetProperty("left", std::format("{:.1f}px", clamped_center - width_px * 0.5f));
            el->SetProperty("width", std::format("{:.1f}px", width_px));
        };

        std::optional<float> strip_hover_x;
        if (film_strip_attached_) {
            if (const auto& hover = film_strip.hoverState(); hover.has_value())
                strip_hover_x = hover->guide_x - timeline_x;
        }

        std::optional<float> hovered_x;
        if (const auto hovered_id = hoveredKeyframeId(); hovered_id.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframeById(*hovered_id))
                hovered_x = timeToX(keyframe->time, 0.0f, timeline_width);
        }

        std::optional<float> selected_x;
        if (const auto selected = controller_.selectedKeyframe(); selected.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframe(*selected))
                selected_x = timeToX(keyframe->time, 0.0f, timeline_width);
        }

        std::optional<float> playhead_x;
        if (playhead_in_range_)
            playhead_x = cached_playhead_screen_x_ - timeline_x;

        set_guide(el_guide_strip_hover_, strip_hover_x, 1.0f);
        set_guide(el_guide_hovered_, hovered_x, 1.5f);
        set_guide(el_guide_selected_, selected_x, 2.0f);
        set_guide(el_guide_playhead_, playhead_x, PLAYHEAD_WIDTH);
    }

    void RmlSequencerPanel::updateTimelineTooltip(const gui::FilmStripRenderer& film_strip,
                                                  const PanelInputState& input) {
        if (!elements_cached_ || !el_timeline_tooltip_) {
            return;
        }

        const auto& hover = film_strip.hoverState();
        if (!film_strip_attached_ || !hover.has_value()) {
            el_timeline_tooltip_->SetProperty("display", "none");
            return;
        }

        std::string html = "<span class=\"timeline-tooltip-line title\">Time ";
        html += formatTime(hover->exact_time);
        html += "</span>";
        if (hover->over_thumbnail) {
            html += "<span class=\"timeline-tooltip-line\">Sample ";
            html += formatTime(hover->sample_time);
            html += "</span>";
            html += "<span class=\"timeline-tooltip-line\">Covers ";
            html += formatTime(hover->interval_start_time);
            html += " - ";
            html += formatTime(hover->interval_end_time);
            html += "</span>";
        }

        const float dp = cached_dp_ratio_;
        const float local_x = input.mouse_x - cached_panel_x_;
        const float local_y = input.mouse_y - cached_panel_y_;
        const float offset_x = 14.0f * dp;
        const float offset_y = 10.0f * dp;
        const bool align_right = local_x > cached_panel_width_ - 180.0f * dp;
        const bool place_below = local_y < 48.0f * dp;

        const float approx_width = 170.0f * dp;
        const float approx_height = hover->over_thumbnail ? 54.0f * dp : 32.0f * dp;
        float left = align_right ? local_x - approx_width - offset_x : local_x + offset_x;
        float top = place_below ? local_y + offset_y : local_y - approx_height - offset_y;
        left = std::clamp(left, 8.0f * dp, std::max(8.0f * dp, cached_panel_width_ - approx_width - 8.0f * dp));
        top = std::clamp(top, 8.0f * dp, std::max(8.0f * dp, cached_total_height_ - approx_height - 8.0f * dp));

        el_timeline_tooltip_->SetInnerRML(html);
        char left_buffer[32];
        char top_buffer[32];
        std::snprintf(left_buffer, sizeof(left_buffer), "%.1fpx", left);
        std::snprintf(top_buffer, sizeof(top_buffer), "%.1fpx", top);
        el_timeline_tooltip_->SetProperty("left", left_buffer);
        el_timeline_tooltip_->SetProperty("top", top_buffer);
        el_timeline_tooltip_->SetProperty("display", "block");
    }

    void RmlSequencerPanel::forwardInput(const PanelInputState& input) {
        if (!rml_context_)
            return;
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(cached_panel_x_ - input.screen_x),
                                            static_cast<int>(cached_panel_y_ - input.screen_y));
        }

        const float local_x = input.mouse_x - cached_panel_x_;
        const float local_y = input.mouse_y - cached_panel_y_;

        const float total_h = cached_total_height_;
        hovered_ = local_x >= 0 && local_y >= 0 &&
                   local_x < cached_panel_width_ && local_y < total_h;

        if (!hovered_) {
            gui::RmlPanelHost::setFrameTooltip({}, nullptr);
            if (last_hovered_)
                rml_context_->ProcessMouseLeave();
            last_hovered_ = false;

            if (input.mouse_clicked[0]) {
                if (auto* const focused = rml_context_->GetFocusElement())
                    focused->Blur();
            }

            auto* const focused = rml_context_->GetFocusElement();
            if (gui::rml_input::hasFocusedKeyboardTarget(focused))
                forwardFocusedKeyboardInput(rml_context_, input);

            wants_keyboard_ = gui::rml_input::hasFocusedKeyboardTarget(focused) ||
                              dragging_playhead_ || dragging_keyframe_ ||
                              controller_.hasSelection() || !selected_keyframes_.empty();
            return;
        }

        last_hovered_ = true;

        rml_context_->ProcessMouseMove(static_cast<int>(local_x),
                                       static_cast<int>(local_y), 0);

        if (input.mouse_clicked[0])
            rml_context_->ProcessMouseButtonDown(0, 0);
        if (!input.mouse_down[0])
            rml_context_->ProcessMouseButtonUp(0, 0);

        auto* hover = rml_context_->GetHoverElement();
        if (hover) {
            gui::RmlPanelHost::setFrameTooltip(gui::resolveRmlTooltip(hover), hover);

            if (input.mouse_clicked[1]) {
                for (auto* el = hover; el; el = el->GetParentNode()) {
                    const auto& id = el->GetId();
                    TransportContextMenuRequest::Target target = TransportContextMenuRequest::Target::NONE;
                    if (id == "btn-snap")
                        target = TransportContextMenuRequest::Target::SNAP;
                    else if (id == "btn-preview")
                        target = TransportContextMenuRequest::Target::PREVIEW;
                    else if (id == "btn-format")
                        target = TransportContextMenuRequest::Target::FORMAT;

                    if (target != TransportContextMenuRequest::Target::NONE) {
                        transport_ctx_request_ = {target, input.mouse_x, input.mouse_y};
                        break;
                    }
                }
            }
        } else {
            gui::RmlPanelHost::setFrameTooltip({}, nullptr);
        }

        auto* const focused = rml_context_->GetFocusElement();
        if (gui::rml_input::hasFocusedKeyboardTarget(focused))
            forwardFocusedKeyboardInput(rml_context_, input);

        wants_keyboard_ = gui::rml_input::hasFocusedKeyboardTarget(focused) ||
                          dragging_playhead_ || dragging_keyframe_ ||
                          controller_.hasSelection() || !selected_keyframes_.empty();
    }

    bool RmlSequencerPanel::consumeSavePathRequest() {
        const bool r = save_path_requested_;
        save_path_requested_ = false;
        return r;
    }

    bool RmlSequencerPanel::consumeLoadPathRequest() {
        const bool r = load_path_requested_;
        load_path_requested_ = false;
        return r;
    }

    bool RmlSequencerPanel::consumeExportRequest() {
        const bool request = export_requested_;
        export_requested_ = false;
        return request;
    }

    bool RmlSequencerPanel::consumeDockToggleRequest() {
        const bool request = dock_toggle_requested_;
        dock_toggle_requested_ = false;
        return request;
    }

    bool RmlSequencerPanel::consumeClosePanelRequest() {
        const bool request = close_panel_requested_;
        close_panel_requested_ = false;
        return request;
    }

    bool RmlSequencerPanel::consumeClearRequest() {
        const bool r = clear_requested_;
        clear_requested_ = false;
        return r;
    }

    void RmlSequencerPanel::updateTransportSettings() {
        if (!elements_cached_)
            return;

        const bool has_camera_keyframes = controller_.timeline().realKeyframeCount() > 0;
        const bool has_any_state = has_camera_keyframes || controller_.timeline().hasAnimationClip();

        if (el_btn_camera_path_)
            el_btn_camera_path_->SetClass("active", ui_state_.show_camera_path);
        if (el_btn_snap_)
            el_btn_snap_->SetClass("active", ui_state_.snap_to_grid);
        if (el_btn_follow_)
            el_btn_follow_->SetClass("active", ui_state_.follow_playback);
        if (el_btn_film_strip_)
            el_btn_film_strip_->SetClass("active", ui_state_.show_film_strip);
        if (el_btn_preview_)
            el_btn_preview_->SetClass("active", ui_state_.show_pip_preview);
        if (el_btn_equirect_)
            el_btn_equirect_->SetClass("active", ui_state_.equirectangular);
        if (el_speed_label_)
            el_speed_label_->SetInnerRML(formatSpeed(ui_state_.playback_speed));
        if (el_format_label_)
            el_format_label_->SetInnerRML(formatPresetShort(ui_state_.preset));
        if (el_resolution_info_) {
            const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
            const bool custom = ui_state_.preset == lfs::io::video::VideoPreset::CUSTOM;
            const int w = custom ? ui_state_.custom_width : info.width;
            const int h = custom ? ui_state_.custom_height : info.height;
            const int fps = custom ? ui_state_.framerate : info.framerate;
            el_resolution_info_->SetInnerRML(std::format("{}x{} @ {}fps", w, h, fps));
        }
        if (!quality_scrub_editing_)
            syncQualityScrub();

        if (el_btn_save_)
            el_btn_save_->SetClass("disabled", !has_camera_keyframes);
        if (el_btn_export_)
            el_btn_export_->SetClass("disabled", !has_camera_keyframes);
        if (el_btn_clear_)
            el_btn_clear_->SetClass("disabled", !has_any_state);
        if (el_panel_) {
            el_panel_->SetClass("is-floating", floating_);
            el_panel_->SetClass("film-strip-attached", film_strip_attached_);
        }
        if (el_floating_header_)
            el_floating_header_->SetClass("hidden", !floating_);
        if (el_film_strip_panel_)
            el_film_strip_panel_->SetProperty("display", film_strip_attached_ ? "block" : "none");
        if (el_transport_dock_sep_)
            el_transport_dock_sep_->SetClass("hidden", false);
        if (el_btn_dock_toggle_) {
            el_btn_dock_toggle_->SetAttribute("data-tooltip",
                                              floating_ ? "tooltip.seq_dock" : "tooltip.seq_undock");
            el_btn_dock_toggle_->SetClass("active", false);
            el_btn_dock_toggle_->SetClass("hidden", false);
        }
        if (el_dock_toggle_label_)
            el_dock_toggle_label_->SetInnerRML(floating_ ? "Dock" : "Undock");
        if (el_btn_close_panel_) {
            el_btn_close_panel_->SetAttribute("data-tooltip", "common.close");
            el_btn_close_panel_->SetClass("hidden", !floating_);
        }
        if (el_close_panel_label_)
            el_close_panel_label_->SetInnerRML(lfs::event::LocalizationManager::getInstance().get("common.close"));
    }

    float RmlSequencerPanel::timelineWidth() const {
        const float s = cached_dp_ratio_;
        return cached_panel_width_ - 2.0f * INNER_PADDING_H * s;
    }

    void RmlSequencerPanel::render(const float panel_x, const float panel_y,
                                   const float panel_width, const float total_height,
                                   const PanelInputState& input,
                                   RenderingManager* rm, SceneManager* sm,
                                   gui::FilmStripRenderer& film_strip) {
        clearPendingComposite();
        const float dp = rml_manager_->getDpRatio();
        cached_dp_ratio_ = dp;

        const float strip_height = film_strip_attached_ ? gui::FilmStripRenderer::STRIP_HEIGHT : 0.0f;
        const float easing_height = EASING_STRIPE_HEIGHT * dp;
        cached_total_height_ = std::max(0.0f, total_height);
        cached_height_ = std::max(0.0f, total_height - easing_height - strip_height);

        cached_panel_x_ = panel_x;
        cached_panel_y_ = panel_y;
        cached_panel_width_ = panel_width;

        const int w = static_cast<int>(panel_width);
        const int h = static_cast<int>(cached_total_height_);

        if (w <= 0 || h <= 0)
            return;

        if (!rml_context_)
            initContext(w, h);
        if (!rml_context_ || !document_)
            return;

        syncTheme();

        const auto& lang = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (lang != last_language_) {
            last_language_ = lang;
            last_keyframe_count_ = static_cast<size_t>(-1);
        }

        if (elements_cached_) {
            el_timeline_->SetProperty("width", std::format("{:.1f}px", timelineWidth()));
        }

        forwardInput(input);

        const float inner_pad_h = INNER_PADDING_H * dp;
        const float inner_pad = INNER_PADDING * dp;
        const float transport_row_h = TRANSPORT_ROW_HEIGHT * dp;
        const float content_height = cached_height_ - 2.0f * inner_pad - transport_row_h;
        const float tl_width = timelineWidth();

        const Vec2 timeline_pos = {panel_x + inner_pad_h,
                                   panel_y + inner_pad + transport_row_h};

        if (elements_cached_) {
            handleTimelineInteraction(timeline_pos, tl_width, content_height, input);
            handleEasingStripeInteraction(timeline_pos.x, tl_width, input);
            rebuildFilmStrip(timeline_pos.x, tl_width,
                             panel_y + cached_height_ + easing_height - BORDER_OVERLAP * dp,
                             input, rm, sm, film_strip);

            cached_playhead_screen_x_ = timeline_pos.x + clampCenteredSpan(
                                                             timeToX(controller_.playhead(), 0.0f, tl_width),
                                                             tl_width,
                                                             PLAYHEAD_HANDLE_WIDTH * dp);
            playhead_in_range_ = cached_playhead_screen_x_ >= timeline_pos.x &&
                                 cached_playhead_screen_x_ <= timeline_pos.x + tl_width;

            updateButtonStates();
            updateTransportSettings();
            updatePlayhead();
            updateTimeDisplay();
            rebuildKeyframes();
            rebuildRuler();
            rebuildEasingStripe(timeline_pos.x, tl_width);
            updateTimelineGuides(timeline_pos.x, tl_width, film_strip);
        }

        if (!rml_manager_->shouldDeferFboUpdate(fbo_)) {
            if (rml_manager_) {
                rml_manager_->trackContextFrame(rml_context_,
                                                static_cast<int>(panel_x - input.screen_x),
                                                static_cast<int>(panel_y - input.screen_y));
            }
            rml_context_->SetDimensions(Rml::Vector2i(w, h));
            rml_context_->Update();

            fbo_.ensure(w, h);
            if (!fbo_.valid())
                return;

            auto* render_iface = rml_manager_->getRenderInterface();
            assert(render_iface);
            render_iface->SetViewport(w, h);

            GLint prev_fbo = 0;
            fbo_.bind(&prev_fbo);
            render_iface->SetTargetFramebuffer(fbo_.fbo());

            if (!floating_) {
                const auto& shell_bg = theme().menu_background();
                glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
            }

            render_iface->BeginFrame();
            rml_context_->Render();
            render_iface->EndFrame();

            render_iface->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (!fbo_.valid())
            return;

        if (floating_) {
            pending_foreground_composite_ = true;
            pending_composite_x_ = panel_x;
            pending_composite_y_ = panel_y;
            pending_composite_width_ = panel_width;
            pending_composite_height_ = cached_total_height_;
            return;
        }

        fbo_.blitToScreen(panel_x, panel_y, panel_width, cached_total_height_,
                          input.screen_w, input.screen_h);
    }

    void RmlSequencerPanel::handleTimelineInteraction(const Vec2& pos, const float width,
                                                      const float height,
                                                      const PanelInputState& input) {
        const float s = cached_dp_ratio_;
        const float timeline_y = pos.y + RULER_HEIGHT * s + 4.0f * s;
        const float timeline_height = height - RULER_HEIGHT * s - 4.0f * s;
        const float bar_half = std::min(timeline_height, TIMELINE_HEIGHT * s) / 2.0f;
        const float y_center = timeline_y + timeline_height / 2.0f;

        const Vec2 bar_min = {pos.x, y_center - bar_half};
        const Vec2 bar_max = {pos.x + width, y_center + bar_half};

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const float mx = input.mouse_x;
        const float my = input.mouse_y;
        const bool mouse_in_timeline = mx >= bar_min.x && mx <= bar_max.x &&
                                       my >= bar_min.y - RULER_HEIGHT * s && my <= bar_max.y;

        if (mouse_in_timeline && !input.want_capture_mouse) {
            const float wheel = input.mouse_wheel;
            if (std::abs(wheel) > 0.01f) {
                if (input.key_ctrl || input.key_super) {
                    const float mouse_time = xToTime(mx, pos.x, width);
                    const float anchor_ratio = std::clamp((mx - pos.x) / width, 0.0f, 1.0f);
                    const float old_zoom = zoom_level_;
                    const float zoom_factor = std::pow(1.0f + ZOOM_SPEED, wheel);
                    zoom_level_ = std::clamp(old_zoom * zoom_factor, MIN_ZOOM, MAX_ZOOM);

                    if (zoom_level_ != old_zoom) {
                        const float new_visible_duration = getDisplayEndTime();
                        pan_offset_ = mouse_time - anchor_ratio * new_visible_duration;
                        clampPanOffset();
                    }
                } else {
                    const float pan_step = std::max(getDisplayEndTime() * 0.12f, 0.1f);
                    pan_offset_ -= wheel * pan_step;
                    clampPanOffset();
                }
            }
        }

        hovered_keyframe_ = std::nullopt;
        const auto& keyframes = timeline.keyframes();
        for (size_t i = 0; i < keyframes.size(); ++i) {
            if (keyframes[i].is_loop_point)
                continue;
            const float x = timeToX(keyframes[i].time, pos.x, width);
            const float dist = std::abs(mx - x);
            const bool hovered = mouse_in_timeline && dist < KEYFRAME_RADIUS * s * 2;
            if (hovered)
                hovered_keyframe_ = i;
        }

        const float playhead_x = pos.x + clampCenteredSpan(
                                             timeToX(controller_.playhead(), 0.0f, width),
                                             width,
                                             PLAYHEAD_HANDLE_WIDTH * s);
        const float playhead_dist = std::abs(mx - playhead_x);
        bool on_playhead_handle = playhead_dist < PLAYHEAD_HIT_RADIUS * s;

        if (on_playhead_handle && hovered_keyframe_.has_value()) {
            const float kf_x = timeToX(keyframes[*hovered_keyframe_].time, pos.x, width);
            if (std::abs(mx - kf_x) < playhead_dist)
                on_playhead_handle = false;
        }

        for (size_t i = 0; i < keyframes.size(); ++i) {
            const bool hovered = hovered_keyframe_.has_value() && *hovered_keyframe_ == i;

            if (hovered && input.mouse_clicked[0] && !on_playhead_handle) {
                const float current_time = input.time;

                if (last_clicked_keyframe_ == i &&
                    (current_time - last_click_time_) < DOUBLE_CLICK_TIME) {
                    editing_keyframe_time_ = true;
                    editing_keyframe_index_ = i;
                    time_edit_buffer_ = std::format("{:.2f}", keyframes[i].time);
                    last_clicked_keyframe_ = std::nullopt;
                } else {
                    last_click_time_ = current_time;
                    last_clicked_keyframe_ = i;

                    if (input.key_shift && controller_.hasSelection()) {
                        const size_t first_sel = *controller_.selectedKeyframe();
                        const size_t lo = std::min(first_sel, i);
                        const size_t hi = std::max(first_sel, i);
                        selected_keyframes_.clear();
                        for (size_t j = lo; j <= hi; ++j) {
                            if (!keyframes[j].is_loop_point)
                                addSelectedKeyframe(selected_keyframes_, keyframes[j].id);
                        }
                    } else if (input.key_ctrl) {
                        const auto id = keyframes[i].id;
                        if (hasSelectedKeyframe(selected_keyframes_, id))
                            removeSelectedKeyframe(selected_keyframes_, id);
                        else
                            addSelectedKeyframe(selected_keyframes_, id);
                    } else {
                        selected_keyframes_.clear();
                        lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = i}.emit();
                        const bool is_first = (i == 0);
                        if (!is_first) {
                            dragging_keyframe_ = true;
                            dragged_keyframe_changed_ = false;
                            dragged_keyframe_id_ = keyframes[i].id;
                            drag_start_mouse_x_ = mx;
                        } else {
                            lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = i}.emit();
                        }
                    }
                }
            }
        }

        if (input.mouse_clicked[0] && mouse_in_timeline && !dragging_keyframe_ &&
            (on_playhead_handle || !hovered_keyframe_.has_value())) {
            dragging_playhead_ = true;
            controller_.beginScrub();
        }
        if (dragging_playhead_) {
            if (input.mouse_down[0]) {
                float time = xToTime(mx, pos.x, width);
                time = std::clamp(time, 0.0f, timeline.endTime());
                if (ui_state_.snap_to_grid)
                    time = snapTime(time);
                controller_.scrub(time);
            } else {
                dragging_playhead_ = false;
                controller_.endScrub();
            }
        }

        if (dragging_keyframe_) {
            if (input.mouse_down[0]) {
                float new_time = xToTime(mx, pos.x, width);
                new_time = std::max(new_time, MIN_KEYFRAME_SPACING);
                if (ui_state_.snap_to_grid)
                    new_time = snapTime(new_time);
                if (controller_.previewKeyframeTimeById(dragged_keyframe_id_, new_time))
                    dragged_keyframe_changed_ = true;
            } else {
                if (dragged_keyframe_changed_)
                    controller_.commitKeyframeTimeById(dragged_keyframe_id_);

                if (!dragged_keyframe_changed_ && std::abs(mx - drag_start_mouse_x_) < DRAG_THRESHOLD_PX) {
                    if (const auto index = controller_.timeline().findKeyframeIndex(dragged_keyframe_id_); index.has_value()) {
                        lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = *index}.emit();
                    }
                }
                dragging_keyframe_ = false;
                const bool emit_keyframe_change = dragged_keyframe_changed_;
                dragged_keyframe_changed_ = false;
                dragged_keyframe_id_ = sequencer::INVALID_KEYFRAME_ID;
                if (emit_keyframe_change)
                    lfs::core::events::state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
            }
        }

        if ((controller_.hasSelection() || !selected_keyframes_.empty()) &&
            input.key_delete_pressed) {
            std::vector<sequencer::KeyframeId> to_delete;
            if (!selected_keyframes_.empty())
                to_delete.assign(selected_keyframes_.begin(), selected_keyframes_.end());
            else if (auto selected_id = controller_.selectedKeyframeId(); selected_id.has_value())
                to_delete.push_back(*selected_id);

            for (const auto& keyframe : keyframes) {
                if (!keyframe.is_loop_point) {
                    std::erase(to_delete, keyframe.id);
                    break;
                }
            }

            bool removed_any = false;
            for (const auto id : to_delete)
                removed_any |= controller_.removeKeyframeById(id);
            for (const auto id : to_delete)
                removeSelectedKeyframe(selected_keyframes_, id);
            if (removed_any)
                lfs::core::events::state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        }

        if (mouse_in_timeline && input.mouse_clicked[1]) {
            context_menu_time_ = std::max(0.0f, xToTime(mx, pos.x, width));
            if (ui_state_.snap_to_grid)
                context_menu_time_ = snapTime(context_menu_time_);
            context_menu_keyframe_ = hovered_keyframe_;
            context_menu_open_ = true;
            context_menu_x_ = mx;
            context_menu_y_ = my;
        }

        // Context menu display is handled by the sequencer RML overlay.
    }

    void RmlSequencerPanel::handleEasingStripeInteraction(const float timeline_x, const float timeline_width,
                                                          const PanelInputState& input) {
        const float dp = cached_dp_ratio_;
        const float stripe_y = cached_panel_y_ + cached_height_;
        const float stripe_h = EASING_STRIPE_HEIGHT * dp;
        const float mx = input.mouse_x;
        const float my = input.mouse_y;

        if (timeline_width <= 0.0f || controller_.timeline().keyframes().empty())
            return;

        if (mx < timeline_x || mx > timeline_x + timeline_width ||
            my < stripe_y || my > stripe_y + stripe_h) {
            return;
        }

        if (input.mouse_clicked[1]) {
            std::optional<size_t> nearest;
            float best_dist = KEYFRAME_RADIUS * 3.0f * dp;
            for (size_t i = 0; i < controller_.timeline().keyframes().size(); ++i) {
                const float dist = std::abs(mx - timeToX(controller_.timeline().keyframes()[i].time,
                                                         timeline_x, timeline_width));
                if (dist < best_dist) {
                    best_dist = dist;
                    nearest = i;
                }
            }

            context_menu_time_ = nearest.has_value()
                                     ? controller_.timeline().keyframes()[*nearest].time
                                     : controller_.playhead();
            context_menu_keyframe_ = nearest;
            context_menu_open_ = true;
        }

        if (input.mouse_clicked[0]) {
            std::optional<size_t> nearest;
            float best_dist = KEYFRAME_RADIUS * 2.0f * dp;
            for (size_t i = 0; i < controller_.timeline().keyframes().size(); ++i) {
                const float dist = std::abs(mx - timeToX(controller_.timeline().keyframes()[i].time,
                                                         timeline_x, timeline_width));
                if (dist < best_dist) {
                    best_dist = dist;
                    nearest = i;
                }
            }
            if (nearest.has_value())
                lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = *nearest}.emit();
        }
    }

    void RmlSequencerPanel::handleFilmStripInteraction(const float timeline_x, const float timeline_width,
                                                       const PanelInputState& input,
                                                       gui::FilmStripRenderer& film_strip) {
        if (!film_strip_attached_) {
            if (film_strip_scrubbing_) {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
            return;
        }

        const bool can_scrub = controller_.timeline().size() >= 2;
        const float scrub_time = can_scrub
                                     ? std::clamp(
                                           sequencer_ui::screenXToTime(input.mouse_x, timeline_x, timeline_width,
                                                                       getDisplayEndTime(), pan_offset_),
                                           controller_.timeline().startTime(), controller_.timeline().endTime())
                                     : 0.0f;

        if (film_strip_scrubbing_) {
            if (input.mouse_down[0] && can_scrub) {
                controller_.scrub(scrub_time);
            } else {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
        }

        if (const auto& hover = film_strip.hoverState(); hover.has_value()) {
            if (!film_strip_scrubbing_ && can_scrub && !input.want_capture_mouse && input.mouse_clicked[0]) {
                film_strip_scrubbing_ = true;
                controller_.beginScrub();
                controller_.scrub(scrub_time);
            }
        }
    }

    void RmlSequencerPanel::openFocalLengthEdit(const size_t index, const float current_focal_mm) {
        editing_focal_length_ = true;
        editing_focal_index_ = index;
        focal_edit_buffer_ = std::format("{:.1f}", current_focal_mm);
    }

    float RmlSequencerPanel::getDisplayEndTime() const {
        return sequencer_ui::displayEndTime(controller_.timeline(), zoom_level_);
    }

    std::optional<sequencer::KeyframeId> RmlSequencerPanel::hoveredKeyframeId() const {
        if (!hovered_keyframe_.has_value())
            return std::nullopt;
        const auto* const keyframe = controller_.timeline().getKeyframe(*hovered_keyframe_);
        if (!keyframe || keyframe->is_loop_point)
            return std::nullopt;
        return keyframe->id;
    }

    void RmlSequencerPanel::clampPanOffset() {
        pan_offset_ = std::clamp(pan_offset_, 0.0f,
                                 sequencer_ui::maxPanOffset(controller_.timeline(), zoom_level_));
    }

    float RmlSequencerPanel::timeToX(const float time, const float timeline_x, const float timeline_width) const {
        return sequencer_ui::timeToScreenX(time, timeline_x, timeline_width, getDisplayEndTime(), pan_offset_);
    }

    float RmlSequencerPanel::xToTime(const float x, const float timeline_x, const float timeline_width) const {
        return sequencer_ui::screenXToTime(x, timeline_x, timeline_width, getDisplayEndTime(), pan_offset_);
    }

    float RmlSequencerPanel::snapTime(const float time) const {
        if (!ui_state_.snap_to_grid || ui_state_.snap_interval <= 0.0f)
            return time;
        return std::round(time / ui_state_.snap_interval) * ui_state_.snap_interval;
    }

    // ── Quality Scrub Field ──────────────────────────────────

    namespace {
        constexpr int QUALITY_MIN = 15;
        constexpr int QUALITY_MAX = 28;
        constexpr float SCRUB_DRAG_THRESHOLD_PX = 4.0f;
    } // namespace

    void RmlSequencerPanel::QualityScrubListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        const auto event_id = event.GetId();
        auto* el = event.GetCurrentElement();

        if (event_id == Rml::EventId::Mousedown && el && el->GetId() == "quality-scrub") {
            const int button = event.GetParameter<int>("button", 0);
            if (button != 0)
                return;
            panel->quality_scrub_active_ = true;
            panel->quality_scrub_dragging_ = false;
            panel->quality_scrub_start_x_ = event.GetParameter<float>("mouse_x", 0.0f);
        } else if (event_id == Rml::EventId::Mousemove && panel->quality_scrub_active_) {
            const float mx = event.GetParameter<float>("mouse_x", 0.0f);
            const float dx = mx - panel->quality_scrub_start_x_;
            if (!panel->quality_scrub_dragging_ && std::abs(dx) < SCRUB_DRAG_THRESHOLD_PX)
                return;

            if (!panel->quality_scrub_dragging_) {
                panel->quality_scrub_dragging_ = true;
                if (panel->el_quality_scrub_)
                    panel->el_quality_scrub_->SetClass("is-dragging", true);
            }
            panel->applyQualityFromDrag(mx);
            event.StopPropagation();
        } else if (event_id == Rml::EventId::Mouseup && panel->quality_scrub_active_) {
            const bool was_dragging = panel->quality_scrub_dragging_;
            panel->quality_scrub_active_ = false;
            panel->quality_scrub_dragging_ = false;
            if (panel->el_quality_scrub_)
                panel->el_quality_scrub_->SetClass("is-dragging", false);

            if (!was_dragging)
                panel->enterQualityEdit();
            event.StopPropagation();
        } else if (event_id == Rml::EventId::Change && el && el->GetId() == "quality-input") {
            const bool linebreak = event.GetParameter<bool>("linebreak", false);
            if (linebreak)
                panel->exitQualityEdit(true);
        } else if (event_id == Rml::EventId::Blur && el && el->GetId() == "quality-input") {
            panel->exitQualityEdit(false);
        }
    }

    void RmlSequencerPanel::syncQualityScrub() {
        if (!el_quality_display_ || !el_quality_fill_)
            return;

        const int value = std::clamp(ui_state_.quality, QUALITY_MIN, QUALITY_MAX);
        const float t = static_cast<float>(value - QUALITY_MIN) /
                        static_cast<float>(QUALITY_MAX - QUALITY_MIN);
        const std::string pct = std::format("{:.1f}%", t * 100.0f);
        el_quality_fill_->SetProperty("width", pct);
        el_quality_display_->SetInnerRML(std::to_string(value));
    }

    void RmlSequencerPanel::applyQualityFromDrag(const float mouse_x) {
        if (!el_quality_scrub_)
            return;

        const float left = el_quality_scrub_->GetAbsoluteLeft();
        const float width = std::max(el_quality_scrub_->GetBox().GetSize().x, 1.0f);
        const float t = std::clamp((mouse_x - left) / width, 0.0f, 1.0f);
        const int value = QUALITY_MIN + static_cast<int>(std::round(t * (QUALITY_MAX - QUALITY_MIN)));
        ui_state_.quality = std::clamp(value, QUALITY_MIN, QUALITY_MAX);
        syncQualityScrub();
    }

    void RmlSequencerPanel::enterQualityEdit() {
        if (!el_quality_scrub_ || !el_quality_input_ || quality_scrub_editing_)
            return;

        quality_scrub_editing_ = true;
        el_quality_scrub_->SetClass("is-editing", true);
        el_quality_input_->SetAttribute("value", std::to_string(ui_state_.quality));
        el_quality_input_->Focus();
    }

    void RmlSequencerPanel::exitQualityEdit(const bool commit) {
        if (!quality_scrub_editing_)
            return;

        if (commit && el_quality_input_) {
            const auto text = el_quality_input_->GetAttribute<Rml::String>("value", "");
            try {
                const int val = std::stoi(text);
                ui_state_.quality = std::clamp(val, QUALITY_MIN, QUALITY_MAX);
            } catch (...) {
            }
        }

        quality_scrub_editing_ = false;
        if (el_quality_scrub_)
            el_quality_scrub_->SetClass("is-editing", false);
        syncQualityScrub();
    }

} // namespace lfs::vis
