/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/timeline_view_math.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fmt/format.h>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;
        constexpr float PLAYHEAD_HIT_RADIUS = 6.0f;
        constexpr float PLAYHEAD_HANDLE_WIDTH = 8.0f;

        [[nodiscard]] std::string formatTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return fmt::format("{}:{:05.2f}", mins, secs);
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

    using namespace panel_config;

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

            segments_html += fmt::format(
                "<div class=\"easing-segment {}\" style=\"left:{:.1f}px;width:{:.1f}px;\"></div>",
                (i % 2 == 0) ? "primary" : "secondary",
                x0, x1 - x0);

            const auto easing = keyframes[i].easing;
            if (easing == sequencer::EasingType::LINEAR) {
                const float len = x1 - x0;
                curves_html += fmt::format(
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
                curves_html += fmt::format(
                    "<div class=\"easing-curve-segment\" style=\"left:{:.1f}px;top:{:.1f}px;width:{:.1f}px;transform:rotate({:.2f}deg);\"></div>",
                    px0, py0, len, angle_deg);
            }
        }

        for (size_t i = 0; i < keyframes.size(); ++i) {
            const float kx = localTimeToX(keyframes[i].time);
            const char* tone = (i % 2 == 0) ? "primary" : "secondary";
            indicators_html += fmt::format(
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

            indicators_html += fmt::format(
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
            divider_html += fmt::format(
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
            sprocket_top_html += fmt::format(
                "<div class=\"film-strip-sprocket top\" style=\"left:{:.1f}px;\"></div>", sx);
            sprocket_bottom_html += fmt::format(
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

                gaps_html += fmt::format(
                    "<div class=\"film-strip-gap\" style=\"left:{:.1f}px;width:{:.1f}px;\">",
                    x_min, x_max - x_min);
                const float stripe_span = gui::FilmStripRenderer::STRIP_HEIGHT -
                                          gui::FilmStripRenderer::THUMB_PADDING * 2.0f;
                for (float stripe_x = -stripe_span; stripe_x < (x_max - x_min) + stripe_span;
                     stripe_x += 10.0f) {
                    gaps_html += fmt::format(
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
                fmt::format("sequencer-film-slot://{}-{}", thumb.slot_idx, texture_id);
            if (render)
                render->register_external_texture(source, texture_id,
                                                  gui::FilmStripRenderer::THUMB_WIDTH,
                                                  gui::FilmStripRenderer::THUMB_HEIGHT,
                                                  true);
            active_sources.insert(source);

            thumb_el->SetProperty("display", "block");
            thumb_el->SetProperty("left", fmt::format("{:.1f}px", thumb.screen_x - groove_origin_x));
            thumb_el->SetProperty("width", fmt::format("{:.1f}px", thumb.screen_width));
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
            markers_html += fmt::format(
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
        el_panel_guides_->SetProperty("left", fmt::format("{:.1f}px", guide_left));
        el_panel_guides_->SetProperty("top", fmt::format("{:.1f}px", guide_top));
        el_panel_guides_->SetProperty("width", fmt::format("{:.1f}px", guide_width));
        el_panel_guides_->SetProperty("height", fmt::format("{:.1f}px", std::max(0.0f, guide_bottom - guide_top)));

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
            el->SetProperty("left", fmt::format("{:.1f}px", clamped_center - width_px * 0.5f));
            el->SetProperty("width", fmt::format("{:.1f}px", width_px));
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
                    time_edit_buffer_ = fmt::format("{:.2f}", keyframes[i].time);
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
        focal_edit_buffer_ = fmt::format("{:.1f}", current_focal_mm);
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

} // namespace lfs::vis
