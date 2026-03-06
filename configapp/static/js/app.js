/**
 * ConfigApp Alpine.js component.
 *
 * Manages wizard state across 5 steps:
 *   Step 1 — Folder scan & file role assignment
 *   Step 2 — Column type + schema configuration (merged)
 *   Step 3 — Stakeholder configuration
 *   Step 4 — Dataset metadata (name, domain, description, citation)
 *   Step 5 — YAML config preview & download
 *
 * Schema (taxonomy category) is configured inline in the Step 2 column table
 * and auto-populates reactively as the user selects column types.
 */

// ─── Taxonomy definitions ───────────────────────────────────────────────────

const TAXONOMY = {
  users: {
    label: "User Data",
    categories: [
      { key: "user_id", label: "User ID", desc: "Primary user identifier", singleton: true, required: true },
      { key: "demographics", label: "Demographics", desc: "Age, gender, occupation, education, location" },
      { key: "downstream_stakeholder_info", label: "Downstream Stakeholder Info", desc: "Social graph data (friends, followers, trust)" },
      { key: "additional_attributes", label: "Additional Attributes", desc: "Other user data (browsing frequency, etc.)" },
      { key: "other", label: "Other", desc: "Does not fit any category above" },
    ],
  },
  items: {
    label: "Item Data",
    categories: [
      { key: "item_id", label: "Item ID", desc: "Primary item identifier", singleton: true, required: true },
      { key: "descriptive_features", label: "Descriptive Features", desc: "Structured: categories, tags, genres" },
      { key: "content_features", label: "Content Features", desc: "Unstructured/semi-structured: text, images" },
      { key: "provider_upstream_info", label: "Provider / Upstream Info", desc: "Creator, seller, author, publisher metadata" },
      { key: "other", label: "Other", desc: "Does not fit any category above" },
    ],
  },
  interactions: {
    label: "Interaction Data",
    categories: [
      { key: "user_id", label: "User ID", desc: "User identifier (links to user data)", singleton: true, required: true },
      { key: "item_id", label: "Item ID", desc: "Item identifier (links to item data)", singleton: true, required: true },
      { key: "explicit_feedback", label: "Explicit Feedback", desc: "Ratings, likes, reviews" },
      { key: "implicit_feedback", label: "Implicit Feedback", desc: "Clicks, purchases, views, play time" },
      { key: "timestamp", label: "Timestamp", desc: "When the interaction occurred", singleton: true },
      { key: "session_data", label: "Session Data", desc: "Session identifiers and metadata" },
      { key: "interaction_context", label: "Interaction Context", desc: "Device, location, other context" },
      { key: "other", label: "Other", desc: "Does not fit any category above" },
    ],
  },
};

// Column name → taxonomy category heuristics
const _NAME_HINTS = [
  { patterns: ["friend", "follow", "social", "trust"], category: "downstream_stakeholder_info" },
  { patterns: ["developer", "publisher", "author", "brand", "seller", "creator", "artist", "winery", "manufacturer"], category: "provider_upstream_info" },
  { patterns: ["age", "gender", "occupation", "education", "location", "country"], category: "demographics" },
  { patterns: ["session"], category: "session_data" },
  { patterns: ["device", "platform", "browser"], category: "interaction_context" },
];

// Type labels for display
const TYPE_LABELS = {
  float: "Number",
  token: "Categorical",
  token_seq: "Cat. list",
  text: "Free text",
  datetime: "Datetime",
  boolean: "Boolean",
  url: "URL",
  misc: "Misc",
  drop: "Exclude",
};


// ─── Main component ─────────────────────────────────────────────────────────

function configApp(defaultPath, outputPath) {
  return {
    // ── Wizard state ──
    step: 1,

    // ── Path configuration ──
    outputPath: outputPath || "",

    // ── Step 1: Folder scan ──
    folderPath: defaultPath || localStorage.getItem("configapp_folder") || "",
    files: [],
    scanning: false,
    scanError: "",

    // ── Step 2: Column config + Schema (merged) ──
    activeTab: "",           // tabKey: "role__filename"
    previews: {},            // { tabKey: previewData }
    columnConfigs: {},       // { tabKey: { colName: { type, separator, schema } } }
    previewLoading: false,
    previewError: "",

    // ── Step 3: Stakeholders ──
    stakeholderConfig: {
      consumer:    { enabled: true,  id_column: "", columns: [] },
      provider:    { enabled: false, id_column: "", columns: [] },
      upstream:    { enabled: false, id_column: "", columns: [] },
      downstream:  { enabled: false, id_column: "", columns: [] },
      system:      { enabled: false, columns: [] },
      third_party: { enabled: false, columns: [] },
    },

    // ── Step 4: Metadata ──
    metadata: {
      datasetName: "",
      domain: "",
      version: "",
      description: "",
      sourceUrl: "",
      citation: "",
    },

    domainOptions: [
      { key: "media_and_entertainment", label: "Media & Entertainment" },
      { key: "news_and_information", label: "News & Information" },
      { key: "publishing_and_literature", label: "Publishing & Literature" },
      { key: "healthcare_and_wellness", label: "Healthcare & Wellness" },
      { key: "education_and_elearning", label: "Education & E-learning" },
      { key: "job_portals_and_career_services", label: "Job Portals & Career Services" },
      { key: "ecommerce_and_retail", label: "E-commerce & Retail" },
      { key: "travel_and_hospitality", label: "Travel & Hospitality" },
      { key: "other", label: "Other" },
    ],

    // ── Step 5: Export ──
    yamlPreview: "",
    yamlLoading: false,
    yamlError: "",
    savedPath: "",

    // ── Computed ──
    get hasInteractions() {
      return this.files.some((f) => f.role === "interactions");
    },

    /** YAML filename: "{name}_{version}.yaml" or "{name}.yaml" if no version. */
    get yamlFilename() {
      const name = this.metadata.datasetName || "dataset";
      const version = (this.metadata.version || "").trim();
      return version ? `${name}_${version}.yaml` : `${name}.yaml`;
    },

    /** Returns list of { role, filename, tabKey } for all assigned files (excluding "skip"). */
    get assignedFiles() {
      const result = [];
      for (const f of this.files) {
        if (f.role && f.role !== "skip") {
          result.push({
            role: f.role,
            filename: f.filename,
            tabKey: f.role + "__" + f.filename,
          });
        }
      }
      return result;
    },

    get currentPreview() {
      return this.previews[this.activeTab] || null;
    },

    /** Get the role for the currently active tab. */
    get activeRole() {
      const idx = this.activeTab.indexOf("__");
      return idx >= 0 ? this.activeTab.slice(0, idx) : this.activeTab;
    },

    /** Get the filename for the currently active tab. */
    get activeFilename() {
      const idx = this.activeTab.indexOf("__");
      return idx >= 0 ? this.activeTab.slice(idx + 2) : "";
    },

    // ── Step 1: Scan folder ──
    async scanFolder() {
      this.scanning = true;
      this.scanError = "";
      this.files = [];
      localStorage.setItem("configapp_folder", this.folderPath);

      try {
        const res = await fetch("/api/scan-folder", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_path: this.folderPath }),
        });
        const data = await res.json();
        if (!res.ok) { this.scanError = data.error || "Scan failed"; return; }
        this.files = data.files.map((f) => ({ ...f, role: "" }));
      } catch (err) {
        this.scanError = "Network error: " + err.message;
      } finally {
        this.scanning = false;
      }
    },

    // ── Step 1 → 2 ──
    async goToStep2() {
      if (!this.hasInteractions) return;
      this.step = 2;
      const files = this.assignedFiles;
      if (files.length > 0) {
        await this.switchTab(files[0].tabKey);
      }
    },

    // ── Step 2: Switch tab & load preview ──
    async switchTab(tabKey) {
      this.activeTab = tabKey;
      this.previewError = "";
      if (this.previews[tabKey]) return;

      // Parse filename from tabKey
      const idx = tabKey.indexOf("__");
      const filename = idx >= 0 ? tabKey.slice(idx + 2) : "";
      if (!filename) return;

      this.previewLoading = true;
      try {
        const res = await fetch("/api/preview-file", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_path: this.folderPath, filename }),
        });
        const data = await res.json();
        if (!res.ok) { this.previewError = data.error || "Preview failed"; return; }
        this.previews[tabKey] = data;

        // Initialise column configs from auto-suggestions (type + separator + schema)
        const role = tabKey.split("__")[0];
        if (!this.columnConfigs[tabKey]) this.columnConfigs[tabKey] = {};
        for (const col of data.columns) {
          const suggestedType = data.suggested_types[col] || "";
          const suggestedSep = data.suggested_separators[col] || "|";
          const suggestedSchema = suggestedType !== "drop"
            ? this._autoSchemaCategory(col, suggestedType, role)
            : "";

          if (!this.columnConfigs[tabKey][col]) {
            // Create new config with all fields
            this.columnConfigs[tabKey][col] = {
              type: suggestedType,
              separator: suggestedSep,
              schema: suggestedSchema,
            };
          } else {
            // Config exists — only set schema if it's empty (user hasn't manually assigned it)
            if (!this.columnConfigs[tabKey][col].schema) {
              this.columnConfigs[tabKey][col].schema = suggestedSchema;
            }
          }
        }
      } catch (err) {
        this.previewError = "Network error: " + err.message;
      } finally {
        this.previewLoading = false;
      }
    },

    // ── Step 2 → 3 (Stakeholders) ──
    goToStep3() {
      if (!this.schemaValid) return;
      this._populateStakeholders();
      this.step = 3;
    },

    // ── Schema auto-population ──────────────────────────────────────────────

    /**
     * Called reactively when the user changes a column's type dropdown.
     * Updates the schema category for that column based on auto-population rules.
     */
    autoUpdateSchema(col) {
      const cfg = this.getColumnConfig(col);
      if (cfg.type === "drop") {
        cfg.schema = "";
        return;
      }
      cfg.schema = this._autoSchemaCategory(col, cfg.type, this.activeRole);
    },

    /**
     * Determine the best taxonomy category for a column based on its type,
     * name, and the file's role. Name-based hints take priority.
     *
     * Returns a category key (e.g. "descriptive_features") or "" if no match.
     */
    _autoSchemaCategory(colName, colType, fileRole) {
      const lc = colName.toLowerCase();

      // ── Name-based hints (highest priority) ──
      for (const hint of _NAME_HINTS) {
        for (const p of hint.patterns) {
          if (lc.includes(p)) {
            const tax = TAXONOMY[fileRole];
            if (tax && tax.categories.some((c) => c.key === hint.category)) {
              return hint.category;
            }
          }
        }
      }

      // ── Type-based rules per role ──

      if (fileRole === "items") {
        if (colType === "text") return "content_features";
        if (colType === "url") return "other";
        if (colType === "misc") return "";  // Doesn't fit any category
        // Everything else left unassigned
      }

      if (fileRole === "interactions") {
        if (colType === "datetime") return "timestamp";
        if (colType === "text") return "explicit_feedback";
        // Everything else left unassigned for interactions
      }

      if (fileRole === "users") {
        if (colType === "boolean") return "additional_attributes";
        // Everything else left unassigned for users
      }

      if (fileRole === "other") {
        // For supplementary files, text → content_features, everything else unassigned
        if (colType === "text") return "content_features";
        if (colType === "url") return "other";
      }

      return "";
    },

    // ── Schema helpers (inline in Step 2) ──────────────────────────────────

    /**
     * Get taxonomy categories available for the active tab's role.
     * For "other" files, return a generic set of categories.
     */
    taxonomyCategoriesFor(tabKey) {
      const role = (tabKey || this.activeTab).split("__")[0];
      if (role === "other") {
        // Generic categories for supplementary files (no required fields)
        return [
          { key: "descriptive_features", label: "Descriptive Features", desc: "Structured metadata", singleton: false, required: false },
          { key: "content_features", label: "Content Features", desc: "Unstructured content", singleton: false, required: false },
          { key: "other", label: "Other", desc: "Does not fit any category", singleton: false, required: false },
        ];
      }
      return TAXONOMY[role]?.categories || [];
    },

    /**
     * Set the schema category for a column, enforcing singleton constraints.
     * Called when the user manually changes the taxonomy dropdown.
     */
    setSchemaForColumn(col, category) {
      const configs = this.columnConfigs[this.activeTab];
      if (!configs) return;

      // If this is a singleton category, unset any other column with the same category
      if (category) {
        const role = this.activeRole;
        const tax = TAXONOMY[role];
        if (tax) {
          const catDef = tax.categories.find((c) => c.key === category);
          if (catDef && catDef.singleton) {
            for (const [otherCol, otherCfg] of Object.entries(configs)) {
              if (otherCfg.schema === category && otherCol !== col) {
                otherCfg.schema = "";
              }
            }
          }
        }
      }
      configs[col].schema = category;
    },

    /** Check if all required schema fields are assigned across all files. */
    get schemaValid() {
      for (const af of this.assignedFiles) {
        const configs = this.columnConfigs[af.tabKey];
        if (!configs) return false;
        const tax = TAXONOMY[af.role];
        if (!tax) continue;
        for (const cat of tax.categories) {
          if (cat.required) {
            const hasAssignment = Object.values(configs).some((c) => c.schema === cat.key);
            if (!hasAssignment) return false;
          }
        }
      }
      return true;
    },

    /** Get validation status for each required category per file tab. */
    getSchemaValidation(tabKey) {
      const role = (tabKey || this.activeTab).split("__")[0];
      const tax = TAXONOMY[role];
      if (!tax) return [];
      const configs = this.columnConfigs[tabKey] || {};
      const results = [];
      for (const cat of tax.categories) {
        if (!cat.required) continue;
        const assignedEntry = Object.entries(configs).find(([, c]) => c.schema === cat.key);
        results.push({
          key: cat.key,
          label: cat.label,
          assigned: !!assignedEntry,
          column: assignedEntry ? assignedEntry[0] : null,
        });
      }
      return results;
    },

    // ── Stakeholder pre-population ─────────────────────────────────────────

    _populateStakeholders() {
      // Consumer: always enabled, find user_id column
      const userIdCol = this._findSchemaColumn("user_id");
      this.stakeholderConfig.consumer.enabled = true;
      this.stakeholderConfig.consumer.id_column = userIdCol || "";
      // Populate consumer columns from demographics + additional_attributes
      this.stakeholderConfig.consumer.columns = [
        ...this._findSchemaColumns("demographics"),
        ...this._findSchemaColumns("additional_attributes"),
      ];

      // Provider: from provider_upstream_info
      const providerCols = this._findSchemaColumns("provider_upstream_info");
      if (providerCols.length > 0) {
        this.stakeholderConfig.provider.enabled = true;
        this.stakeholderConfig.provider.columns = [...providerCols];
        this.stakeholderConfig.provider.id_column = "";
      }

      // Downstream: from downstream_stakeholder_info
      const downstreamCols = this._findSchemaColumns("downstream_stakeholder_info");
      if (downstreamCols.length > 0) {
        this.stakeholderConfig.downstream.enabled = true;
        this.stakeholderConfig.downstream.columns = [...downstreamCols];
        this.stakeholderConfig.downstream.id_column = "";
      }
    },

    /** Find the first column assigned to a given schema category (across all files). */
    _findSchemaColumn(category) {
      for (const af of this.assignedFiles) {
        const configs = this.columnConfigs[af.tabKey] || {};
        for (const [col, cfg] of Object.entries(configs)) {
          if (cfg.schema === category) return col;
        }
      }
      return null;
    },

    /** Find ALL columns assigned to a given schema category. */
    _findSchemaColumns(category) {
      const cols = [];
      for (const af of this.assignedFiles) {
        const configs = this.columnConfigs[af.tabKey] || {};
        for (const [col, cfg] of Object.entries(configs)) {
          if (cfg.schema === category) cols.push(col);
        }
      }
      return cols;
    },

    /** Get all non-excluded columns across all files (for stakeholder dropdowns). */
    get allAvailableColumns() {
      const cols = [];
      const seen = new Set();
      for (const af of this.assignedFiles) {
        const preview = this.previews[af.tabKey];
        const configs = this.columnConfigs[af.tabKey] || {};
        if (!preview) continue;
        for (const col of preview.columns) {
          const cfg = configs[col] || {};
          if (cfg.type === "drop") continue;
          if (!seen.has(col)) {
            seen.add(col);
            cols.push({ name: col, type: cfg.type || "", role: af.role, filename: af.filename });
          }
        }
      }
      return cols;
    },

    /** Toggle a column in a stakeholder's columns list. */
    toggleStakeholderColumn(stakeholder, colName) {
      const cfg = this.stakeholderConfig[stakeholder];
      const idx = cfg.columns.indexOf(colName);
      if (idx >= 0) {
        cfg.columns.splice(idx, 1);
      } else {
        cfg.columns.push(colName);
      }
    },

    // ── Step 2 Helpers ──────────────────────────────────────────────────────

    getColumnConfig(col) {
      if (!this.columnConfigs[this.activeTab]) this.columnConfigs[this.activeTab] = {};
      if (!this.columnConfigs[this.activeTab][col]) {
        this.columnConfigs[this.activeTab][col] = { type: "", separator: "|", schema: "" };
      }
      return this.columnConfigs[this.activeTab][col];
    },

    /**
     * Get sample values for a column as structured objects.
     * Returns: [{ display, isList, parts }]
     */
    getSamples(col) {
      const preview = this.currentPreview;
      if (!preview || !preview.rows.length) return [];
      const idx = preview.columns.indexOf(col);
      if (idx < 0) return [];

      const cfg = this.getColumnConfig(col);
      const seen = new Set();
      const samples = [];

      for (const row of preview.rows) {
        const raw = row[idx];
        const str = String(raw);
        if (seen.has(str) || str === "None" || str === "null" || str === "") continue;
        seen.add(str);

        const sample = {
          display: str.length > 40 ? str.slice(0, 37) + "\u2026" : str,
          isList: false,
          parts: [],
        };

        // Detect list values
        if (cfg.type === "token_seq") {
          const sep = cfg.separator || "|";
          if (str.includes(sep)) {
            sample.isList = true;
            sample.parts = str.split(sep).map((p) => p.trim()).filter(Boolean).slice(0, 5);
          }
        }
        // Also detect Python list repr: ['a', 'b', 'c']
        if (!sample.isList && str.startsWith("[") && str.endsWith("]")) {
          try {
            // Try parsing as JSON first
            const parsed = JSON.parse(str.replace(/'/g, '"'));
            if (Array.isArray(parsed) && parsed.length > 1) {
              sample.isList = true;
              sample.parts = parsed.map(String).slice(0, 5);
            }
          } catch { /* not valid JSON, leave as-is */ }
        }

        samples.push(sample);
        if (samples.length >= 3) break;
      }
      return samples;
    },

    rowClass(col) {
      const cfg = this.getColumnConfig(col);
      const t = cfg.type;
      if (t === "float") return "row-float";
      if (t === "token" || t === "token_seq") return "row-token";
      if (t === "text") return "row-text";
      if (t === "datetime") return "row-datetime";
      if (t === "boolean") return "row-boolean";
      if (t === "url") return "row-url";
      if (t === "misc") return "row-misc";
      if (t === "drop") return "row-drop";
      return "";
    },

    /** Get a human-readable type label. */
    typeLabel(typeKey) {
      return TYPE_LABELS[typeKey] || typeKey || "\u2014";
    },

    tabLabel(af) {
      // Show "Role: filename" when there are multiple files with same role
      const sameRole = this.assignedFiles.filter((f) => f.role === af.role);
      if (sameRole.length > 1) {
        return this.capitalize(af.role) + ": " + af.filename;
      }
      return this.capitalize(af.role);
    },

    capitalize(s) {
      return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
    },

    stakeholderLabel(key) {
      const labels = {
        consumer: "User (Consumer)",
        provider: "Provider",
        upstream: "Upstream",
        downstream: "Downstream",
        system: "System",
        third_party: "Third-party",
      };
      return labels[key] || key;
    },

    isEntityStakeholder(key) {
      return ["consumer", "provider", "upstream", "downstream"].includes(key);
    },

    // ── Step 3 → 4 ──────────────────────────────────────────────────────────

    goToStep4() {
      // Auto-populate dataset name from folder path if empty
      if (!this.metadata.datasetName) {
        const parts = this.folderPath.replace(/\/+$/, "").split("/");
        this.metadata.datasetName = (parts[parts.length - 1] || "")
          .toLowerCase().replace(/\s+/g, "_");
      }
      this.step = 4;
    },

    // ── Step 4 → 5 ──────────────────────────────────────────────────────────

    async goToStep5() {
      if (!this.metadata.datasetName || !this.metadata.domain) return;
      this.step = 5;
      await this.generateYamlPreview();
    },

    // ── Step 5: YAML generation & download ──────────────────────────────────

    async generateYamlPreview() {
      this.yamlLoading = true;
      this.yamlError = "";
      this.yamlPreview = "";
      try {
        const res = await fetch("/api/export-config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this._buildExportPayload()),
        });
        const data = await res.json();
        if (!res.ok) {
          this.yamlError = data.error || "Export failed";
          return;
        }
        this.yamlPreview = data.yaml;
      } catch (err) {
        this.yamlError = "Network error: " + err.message;
      } finally {
        this.yamlLoading = false;
      }
    },

    _buildExportPayload() {
      return {
        metadata: { ...this.metadata },
        files: this.assignedFiles.map((af) => ({
          role: af.role,
          filename: af.filename,
        })),
        columnConfigs: JSON.parse(JSON.stringify(this.columnConfigs)),
        stakeholderConfig: JSON.parse(JSON.stringify(this.stakeholderConfig)),
      };
    },

    downloadYaml() {
      if (!this.yamlPreview) return;
      const blob = new Blob([this.yamlPreview], { type: "text/yaml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = this.yamlFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },

    async saveYaml() {
      if (!this.yamlPreview || !this.outputPath) return;
      this.savedPath = "";
      this.yamlError = "";
      const filename = this.yamlFilename;
      try {
        const res = await fetch("/api/save-config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ yaml: this.yamlPreview, filename }),
        });
        const data = await res.json();
        if (!res.ok) {
          this.yamlError = data.error || "Save failed";
          return;
        }
        this.savedPath = data.saved_to;
      } catch (err) {
        this.yamlError = "Network error: " + err.message;
      }
    },
  };
}
